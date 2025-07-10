import re
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

from loguru import logger

from swe_care.schema.collect import LabeledReviewComment, ReviewCommentLabels
from swe_care.utils.github import GitHubAPI
from swe_care.utils.patch import is_line_changed_in_patch


def extract_problem_statement(closing_issues: list[dict[str, Any]]) -> str:
    """Extract problem statement from closing issues."""
    if not closing_issues:
        return ""

    problem_parts = []
    for issue in closing_issues:
        title = issue.get("title", "")
        body = issue.get("body", "")
        if title:
            problem_parts.append(f"{title}\n{body}\n")

    return "\n".join(problem_parts).strip()


def extract_hints(
    pr_data: dict[str, Any], commit_to_review: Optional[str] = None
) -> str:
    """Extract hints from issues associated with the pull request before the given commit.

    Args:
        pr_data: PR data containing commits and closing issues
        commit_to_review: Optional commit OID to filter comments before. If not given,
                         uses the first commit of the PR.

    Returns:
        Aggregated comment bodies from issues, filtered by commit date.
    """
    # Get all commits from the PR
    commits = pr_data.get("commits", {}).get("nodes", [])
    if not commits:
        return ""

    # Find the target commit
    target_commit = None
    if commit_to_review:
        # Find the specific commit by OID
        for commit_node in commits:
            commit = commit_node.get("commit", {})
            if commit.get("oid") == commit_to_review:
                target_commit = commit
                break
    else:
        # Use the first commit
        target_commit = commits[0].get("commit", {})

    if not target_commit:
        return ""

    # Get the commit date
    commit_date_str = target_commit.get("committedDate", "")
    if not commit_date_str:
        return ""

    try:
        # Parse the commit date (ISO format)
        commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return ""

    # Get all closing issues and their comments
    closing_issues = pr_data.get("closingIssuesReferences", {}).get("nodes", [])
    hints_parts = []

    for issue in closing_issues:
        comments = issue.get("comments", {}).get("nodes", [])
        for comment in comments:
            comment_updated_str = comment.get("updatedAt", "")
            comment_body = comment.get("body", "")

            if not comment_updated_str or not comment_body:
                continue

            try:
                # Parse the comment update date
                comment_date = datetime.fromisoformat(
                    comment_updated_str.replace("Z", "+00:00")
                )

                # Only include comments updated before the commit date
                if comment_date < commit_date:
                    hints_parts.append(comment_body)
            except (ValueError, AttributeError):
                continue

    return "\n".join(hints_parts).strip()


def extract_labeled_review_comments_by_commit(
    pr_data: dict[str, Any],
    commit_to_review: str,
    merged_commit: str,
    repo: str,
    tokens: Optional[list[str]] = None,
) -> list[LabeledReviewComment]:
    """
    Extract and label review comments for a given commit.

    Args:
        pr_data: PR data containing reviews and review threads
        commit_to_review: The commit OID to filter comments for
        merged_commit: The final merged commit
        repo: Repository name in format 'owner/repo'
        tokens: GitHub API tokens

    Returns:
        List of labeled review comments for the specified commit.
    """
    labeled_comments = []

    # Get all review threads
    review_threads = pr_data.get("reviewThreads", {}).get("nodes", [])
    review_threads_with_comment_ids = defaultdict(list)

    # Collect IDs of comments in threads, mapping thread ID to list of comment IDs
    for thread in review_threads:
        thread_comments = thread.get("comments", {}).get("nodes", [])
        for comment in thread_comments:
            comment_id = comment.get("id")
            if comment_id:
                review_threads_with_comment_ids[thread.get("id")].append(comment_id)

    # Get all review comments from all reviews
    reviews = pr_data.get("reviews", {}).get("nodes", [])
    comment_id_to_comment = {}

    for review in reviews:
        review_comments = review.get("comments", {}).get("nodes", [])
        for comment in review_comments:
            comment_id = comment.get("id")
            if comment_id:
                comment_id_to_comment[comment_id] = comment

    # Group comments by thread for review threads with matching commit
    thread_comments_map = defaultdict(list)

    for thread_id, comment_ids in review_threads_with_comment_ids.items():
        for comment_id in comment_ids:
            comment = comment_id_to_comment.get(comment_id)
            if comment:
                # Check if the comment's originalCommit matches commit_to_review
                original_commit = comment.get("originalCommit", {})
                if original_commit and original_commit.get("oid") == commit_to_review:
                    thread_comments_map[thread_id].append(comment)

    # Get PR author login for comparison
    pr_author_login = pr_data.get("author", {}).get("login", "")

    # Get patch for line change detection
    patch_content = ""
    try:
        patch_content = fetch_patch_between_commits(
            repo=repo,
            base_commit=commit_to_review,
            head_commit=merged_commit,
            tokens=tokens,
        )
    except Exception as e:
        logger.warning(
            f"Failed to get patch for {commit_to_review} -> {merged_commit}: {e}"
        )

    # Create LabeledReviewComment for each review thread with matching comments
    for thread_id, comments in thread_comments_map.items():
        if not comments:
            continue

        # Sort comments by createdAt date
        comments.sort(key=lambda c: c.get("createdAt", ""))

        # Aggregate comment bodies with user differentiation
        aggregated_parts = []
        user_mapping = {}  # Maps actual username to numbered identifier
        user_counter = 1  # Counter for assigning user numbers

        for comment in comments:
            comment_body = comment.get("body", "")
            if comment_body:
                # Replace @mentions in comment body with censored usernames
                def replace_mention(match):
                    nonlocal user_counter
                    mentioned_username = match.group(1)

                    # Use @author if it's the PR author
                    if mentioned_username == pr_author_login:
                        return "@author"
                    else:
                        # For non-author users, assign or reuse numbered identifier
                        if mentioned_username not in user_mapping:
                            if mentioned_username:  # Valid username
                                user_mapping[mentioned_username] = (
                                    f"@user{user_counter}"
                                )
                                user_counter += 1
                            else:  # Empty or missing username (shouldn't happen in this context)
                                user_mapping[mentioned_username] = "@unknown"
                        return user_mapping[mentioned_username]

                # Replace all @mentions in the comment body
                comment_body = re.sub(r"@(\w+)", replace_mention, comment_body)

                # Get comment author login
                comment_author = comment.get("author", {})
                if comment_author:
                    author_login = comment_author.get("login", "")

                    # Use @author if it's the PR author
                    if author_login == pr_author_login:
                        user_identifier = "@author"
                    else:
                        # For non-author users, assign or reuse numbered identifier
                        if author_login not in user_mapping:
                            if author_login:  # Valid username
                                user_mapping[author_login] = f"@user{user_counter}"
                                user_counter += 1
                            else:  # Empty or missing username
                                user_mapping[author_login] = "@unknown"
                        user_identifier = user_mapping[author_login]

                    aggregated_parts.append(f"{user_identifier}\n{comment_body}")
                else:
                    # Fallback if author information is missing
                    aggregated_parts.append(f"@unknown\n{comment_body}")

        aggregated_text = "\n".join(aggregated_parts)

        # Use the first comment for other fields since they should be the same for the thread
        first_comment = comments[0]

        # Find the corresponding thread to get additional metadata
        corresponding_thread = None
        for thread in review_threads:
            if thread.get("id") == thread_id:
                corresponding_thread = thread
                break

        if corresponding_thread and aggregated_text:
            # Extract thread metadata for labels
            is_resolved = corresponding_thread.get("isResolved", False)
            is_outdated = corresponding_thread.get("isOutdated", False)
            is_collapsed = corresponding_thread.get("isCollapsed", False)

            # Determine the line number to check for line change detection
            line_to_check = None
            if first_comment.get("line") is not None:
                line_to_check = first_comment.get("line")
            elif first_comment.get("originalLine") is not None:
                line_to_check = first_comment.get("originalLine")
            elif first_comment.get("startLine") is not None:
                line_to_check = first_comment.get("startLine")
            elif first_comment.get("originalStartLine") is not None:
                line_to_check = first_comment.get("originalStartLine")

            # Check if referenced line was changed in merged commit
            referenced_line_changed = False
            if (
                line_to_check is not None
                and first_comment.get("path")
                and patch_content
            ):
                try:
                    referenced_line_changed = is_line_changed_in_patch(
                        patch_content=patch_content,
                        file_path=first_comment.get("path"),
                        line_number=line_to_check,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to check line change for {first_comment.get('path')}:{line_to_check}: {e}"
                    )

            labeled_comment = LabeledReviewComment(
                text=aggregated_text,
                path=first_comment.get("path", ""),
                diff_hunk=first_comment.get("diffHunk"),
                line=first_comment.get("line"),
                start_line=first_comment.get("startLine"),
                original_line=first_comment.get("originalLine"),
                original_start_line=first_comment.get("originalStartLine"),
                labels=ReviewCommentLabels(
                    referenced_line_changed_in_merged_commit=referenced_line_changed,
                    is_resolved=is_resolved,
                    is_outdated=is_outdated,
                    is_collapsed=is_collapsed,
                ),
            )
            labeled_comments.append(labeled_comment)

    return labeled_comments


def fetch_patch_between_commits(
    repo: str, base_commit: str, head_commit: str, tokens: Optional[list[str]] = None
) -> str:
    """Extract patch between two commits."""
    github_api = GitHubAPI(tokens=tokens)
    return github_api.get_patch(repo, base_commit=base_commit, head_commit=head_commit)


def fetch_pr_patch(
    repo: str, pull_number: int, tokens: Optional[list[str]] = None
) -> str:
    """Extract patch for the entire PR."""
    github_api = GitHubAPI(tokens=tokens)
    return github_api.get_patch(repo, pr_number=pull_number)


def fetch_repo_language(repo: str, tokens: Optional[list[str]] = None) -> str:
    """Get the language of a repository."""

    # To fix unhashable type: 'list' error thrown by lru_cache
    @lru_cache(maxsize=128)
    def _get_repo_language(repo: str, tokens: Optional[tuple[str, ...]] = None) -> str:
        """Get the language of a repository."""
        github_api = GitHubAPI(tokens=tokens)

        # Try to get repository info first
        response = github_api.call_api(f"repos/{repo}")
        repo_data = response.json()

        if "language" in repo_data and repo_data["language"]:
            return repo_data["language"]
        else:
            logger.warning(
                f"Repository {repo} does not have a language field, trying to get primary language"
            )
            # Get languages endpoint
            response = github_api.call_api(f"repos/{repo}/languages")
            languages = response.json()

            if languages:
                return max(languages, key=languages.get)
            else:
                raise Exception(f"Repository {repo} has no languages: {languages}")

    if tokens:
        tokens = tuple(tokens)

    return _get_repo_language(repo, tokens)

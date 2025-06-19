from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

from loguru import logger

from swe_reason_bench.schema.dataset import ReferenceReviewComment
from swe_reason_bench.utils.github import GitHubAPI


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


def extract_reference_review_comments(
    pr_data: dict[str, Any], commit_to_review: str
) -> list[ReferenceReviewComment]:
    """Extract reference review comments that are resolved for the given commit.

    Args:
        pr_data: PR data containing reviews and review threads
        commit_to_review: The commit OID to filter comments for

    Returns:
        List of resolved review comments for the specified commit.
    """
    reference_review_comments = []

    # Get all review threads that are resolved
    review_threads = pr_data.get("reviewThreads", {}).get("nodes", [])
    resolved_comment_ids = set()

    # Collect IDs of comments in resolved threads
    for thread in review_threads:
        if thread.get("isResolved", False):
            thread_comments = thread.get("comments", {}).get("nodes", [])
            for comment in thread_comments:
                comment_id = comment.get("id")
                if comment_id:
                    resolved_comment_ids.add(comment_id)

    # Get all review comments and filter by resolved status and commit
    reviews = pr_data.get("reviews", {}).get("nodes", [])
    for review in reviews:
        review_comments = review.get("comments", {}).get("nodes", [])
        for comment in review_comments:
            comment_id = comment.get("id")
            original_commit_oid = comment.get("originalCommit", {}).get("oid")

            # Only include comments that are:
            # 1. In resolved threads
            # 2. Associated with the commit we're reviewing
            if (
                comment_id in resolved_comment_ids
                and original_commit_oid == commit_to_review
            ):
                reference_review_comments.append(
                    ReferenceReviewComment(
                        body=comment.get("body", ""),
                        created_at=comment.get("createdAt", ""),
                        updated_at=comment.get("updatedAt", ""),
                        path=comment.get("path", ""),
                        diff_hunk=comment.get("diffHunk", ""),
                        line=comment.get("line", 0),
                        start_line=comment.get("startLine", 0),
                        original_line=comment.get("originalLine", 0),
                        original_start_line=comment.get("originalStartLine", 0),
                    )
                )

    return reference_review_comments


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

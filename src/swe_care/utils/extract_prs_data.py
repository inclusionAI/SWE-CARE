import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from functools import lru_cache, wraps
from typing import Any, Optional

from loguru import logger

from swe_care.schema.collect import LabeledReviewComment, ReviewCommentLabels
from swe_care.utils.github import GitHubAPI
from swe_care.utils.patch import is_line_changed_in_patch
from swe_care.utils.retrieval import (
    DOCUMENT_ENCODING_FUNCTIONS,
    clone_repo,
    make_index,
    search,
)


def cached_with_tuple_conversion(func):
    """Decorator that converts list arguments to tuples for lru_cache compatibility."""

    @lru_cache(maxsize=128)
    def cached_func(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert list arguments to tuples
        converted_args = []
        for arg in args:
            if isinstance(arg, list):
                converted_args.append(tuple(arg) if arg else None)
            else:
                converted_args.append(arg)

        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                converted_kwargs[key] = tuple(value) if value else None
            else:
                converted_kwargs[key] = value

        return cached_func(*converted_args, **converted_kwargs)

    return wrapper


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


def is_referenced_line_changed_in_merged_commit(
    repo: str,
    commit_to_review: str,
    file_path: str,
    line: int,
    merged_commit: str,
    tokens: Optional[list[str]] = None,
) -> bool:
    """
    Check if a referenced line was changed in the merged commit.

    Args:
        repo: Repository name in format 'owner/repo'
        commit_to_review: The commit OID being reviewed
        file_path: Path to the file
        line: Line number to check
        merged_commit: The final merged commit
        tokens: GitHub API tokens

    Returns:
        True if the referenced line was changed in the merged commit
    """
    try:
        # Fetch patch between commits
        patch_content = fetch_patch_between_commits(
            repo=repo,
            base_commit=commit_to_review,
            head_commit=merged_commit,
            tokens=tokens,
        )

        # Check if the line was changed
        return is_line_changed_in_patch(
            patch_content=patch_content,
            file_path=file_path,
            line_number=line,
        )
    except Exception as e:
        logger.warning(f"Failed to check line change for {file_path}:{line}: {e}")
        return False


def anonymize_comment_body(
    comment_body: str,
    pr_author_login: str,
    user_mapping: dict[str, str],
    user_counter: dict[str, int],
) -> str:
    """
    Anonymize @mentions in comment body.

    Args:
        comment_body: The original comment body
        pr_author_login: The PR author's login name
        user_mapping: Mapping of actual username to anonymized identifier
        user_counter: Counter for assigning user numbers (passed as dict to allow mutation)

    Returns:
        Comment body with anonymized @mentions
    """

    def replace_mention(match):
        mentioned_username = match.group(1)

        # Use @author if it's the PR author
        if mentioned_username == pr_author_login:
            return "@author"
        else:
            # For non-author users, assign or reuse numbered identifier
            if mentioned_username not in user_mapping:
                if mentioned_username:  # Valid username
                    user_mapping[mentioned_username] = f"@user{user_counter['count']}"
                    user_counter["count"] += 1
                else:  # Empty or missing username
                    user_mapping[mentioned_username] = "@unknown"
            return user_mapping[mentioned_username]

    # Replace all @mentions in the comment body
    return re.sub(r"@(\w+)", replace_mention, comment_body)


def _extract_thread_comments_data(
    pr_data: dict[str, Any],
    commit_to_review: str,
    pr_commit_oids: set[str],
    commits: list[dict[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """
    Extract thread comments data organized by thread ID.

    Args:
        pr_data: PR data containing reviews and review threads
        commit_to_review: The commit OID to filter comments for
        pr_commit_oids: Set of all commit OIDs in the PR
        commits: List of all commits in the PR with metadata

    Returns:
        Dict mapping thread_id to (comments_list, thread_metadata)
    """
    # Get all review threads
    review_threads = pr_data.get("reviewThreads", {}).get("nodes", [])
    thread_id_to_metadata = {thread.get("id"): thread for thread in review_threads}

    # Collect IDs of comments in threads
    review_threads_with_comment_ids = defaultdict(list)
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
    thread_comments_data = {}

    for thread_id, comment_ids in review_threads_with_comment_ids.items():
        relevant_comments = []

        for comment_id in comment_ids:
            comment = comment_id_to_comment.get(comment_id)
            if comment:
                should_include_comment = _should_include_comment_for_commit(
                    comment,
                    commit_to_review,
                    pr_commit_oids,
                    commits,
                )

                if should_include_comment:
                    relevant_comments.append(comment)

        if relevant_comments:
            thread_metadata = thread_id_to_metadata.get(thread_id, {})
            thread_comments_data[thread_id] = (relevant_comments, thread_metadata)

    return thread_comments_data


def _should_include_comment_for_commit(
    comment: dict[str, Any],
    commit_to_review: str,
    pr_commit_oids: set[str],
    commits: list[dict[str, Any]],
) -> bool:
    """
    Determine if a review comment should be included for a specific commit.

    This function handles the complex scenario where GitHub's originalCommit
    becomes orphaned due to commit history rewriting (force pushes, amends, etc.).

    GitHub Behavior:
    - When a review comment is made on a commit, GitHub stores both `originalCommit`
      and `commit` fields
    - `originalCommit`: The commit where the comment was originally made
    - `commit`: The current commit (often updated to merge commit or latest commit)
    - When commits are force-pushed or amended, `originalCommit` may become orphaned
      (not present in the PR's commit list) while `commit` gets updated

    Strategies:
    1. Direct originalCommit matching (ideal case)
    2. Temporal matching for orphaned commits: A comment belongs to commit A if it was
       created after commit A's committedDate but before the subsequent commit B's committedDate

    Args:
        comment: The review comment data
        commit_to_review: The target commit OID to match against
        pr_commit_oids: Set of all commit OIDs in the PR
        commits: List of all commits in the PR with metadata

    Returns:
        True if the comment should be included for the commit
    """
    # Strategy 1: Check if the comment's originalCommit matches commit_to_review
    original_commit = comment.get("originalCommit", {})
    original_commit_oid = original_commit.get("oid") if original_commit else None

    if original_commit_oid == commit_to_review:
        logger.trace(f"Comment matched by originalCommit: {original_commit_oid}")
        return True

    # Strategy 2: Handle orphaned originalCommit with temporal matching
    # If originalCommit is not in PR commits, use temporal logic
    if original_commit_oid and original_commit_oid not in pr_commit_oids:
        try:
            comment_created_str = comment.get("createdAt", "")
            if not comment_created_str:
                return False

            comment_created = datetime.fromisoformat(
                comment_created_str.replace("Z", "+00:00")
            )

            # Sort commits by committedDate to ensure chronological order
            sorted_commits = []
            for commit_node in commits:
                commit = commit_node.get("commit", {})
                commit_date_str = commit.get("committedDate", "")
                if commit_date_str and commit.get("oid"):
                    commit_date = datetime.fromisoformat(
                        commit_date_str.replace("Z", "+00:00")
                    )
                    sorted_commits.append((commit.get("oid"), commit_date))

            sorted_commits.sort(key=lambda x: x[1])  # Sort by date

            # Find the target commit and its position
            target_commit_index = None
            target_commit_date = None

            for i, (commit_oid, commit_date) in enumerate(sorted_commits):
                if commit_oid == commit_to_review:
                    target_commit_index = i
                    target_commit_date = commit_date
                    break

            if target_commit_date is None:
                return False

            # Check if comment was created after the target commit
            if comment_created <= target_commit_date:
                return False

            # Check if there's a subsequent commit
            if target_commit_index is not None and target_commit_index + 1 < len(
                sorted_commits
            ):
                # Get the next commit's date
                _, next_commit_date = sorted_commits[target_commit_index + 1]

                # Comment should be between target commit and next commit
                if comment_created < next_commit_date:
                    logger.trace(
                        f"Comment matched by temporal logic: created between {commit_to_review} and next commit"
                    )
                    return True
            else:
                # This is the last commit, so comment should be after target commit
                logger.trace(
                    f"Comment matched by temporal logic: created after last commit {commit_to_review}"
                )
                return True

        except (ValueError, AttributeError) as e:
            logger.trace(f"Failed to parse dates for temporal matching: {e}")

    return False


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

    # Get all commit OIDs from the PR for reference 获得所有PR下的commit ID
    pr_commit_oids = set()
    commits = pr_data.get("commits", {}).get("nodes", [])
    for commit_node in commits:
        commit = commit_node.get("commit", {})
        if commit.get("oid"):
            pr_commit_oids.add(commit.get("oid"))

    # Extract thread comments data using helper function
    thread_comments_data = _extract_thread_comments_data(
        pr_data, commit_to_review, pr_commit_oids, commits
    )

    # Get PR author login for comparison
    pr_author_login = pr_data.get("author", {}).get("login", "")

    # Create LabeledReviewComment for each review thread with matching comments
    for thread_id, (comments, thread_metadata) in thread_comments_data.items():
        if not comments:
            continue

        # Sort comments by createdAt date
        comments.sort(key=lambda c: c.get("createdAt", ""))

        # Aggregate comment bodies with user differentiation
        aggregated_parts = []
        user_mapping = {}  # Maps actual username to numbered identifier
        user_counter = {"count": 1}  # Counter for assigning user numbers

        for comment in comments:
            comment_body = comment.get("body", "")
            if comment_body:
                # Anonymize @mentions in comment body
                comment_body = anonymize_comment_body(
                    comment_body, pr_author_login, user_mapping, user_counter
                )

                # Get comment author login and anonymize
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
                                user_mapping[author_login] = (
                                    f"@user{user_counter['count']}"
                                )
                                user_counter["count"] += 1
                            else:  # Empty or missing username
                                user_mapping[author_login] = "@unknown"
                        user_identifier = user_mapping[author_login]

                    aggregated_parts.append(f"{user_identifier}:\n{comment_body}\n")
                else:
                    # Fallback if author information is missing
                    aggregated_parts.append(f"@unknown:\n{comment_body}\n")

        aggregated_text = "\n".join(aggregated_parts).strip()

        if aggregated_text:
            # Use the first comment for other fields since they should be the same for the thread
            first_comment = comments[0]

            # Extract thread metadata for labels
            is_resolved = thread_metadata.get("isResolved", False)
            is_outdated = thread_metadata.get("isOutdated", False)
            is_collapsed = thread_metadata.get("isCollapsed", False)

            # Check if any comment in the thread is marked as dismissed
            # A comment is dismissed if it's minimized for reasons other than being resolved
            marked_as_dismissed = False
            for comment in comments:
                is_minimized = comment.get("isMinimized", False)
                minimized_reason = comment.get("minimizedReason") or ""
                minimized_reason = minimized_reason.upper()

                if is_minimized and minimized_reason != "RESOLVED":
                    marked_as_dismissed = True
                    break

            # Determine the line number to check for line change detection
            line_to_check = None
            if first_comment.get("originalLine") is not None:
                line_to_check = first_comment.get("originalLine")
            elif first_comment.get("line") is not None:
                line_to_check = first_comment.get("line")
            elif first_comment.get("startLine") is not None:
                line_to_check = first_comment.get("startLine")
            elif first_comment.get("originalStartLine") is not None:
                line_to_check = first_comment.get("originalStartLine")

            # Check if referenced line was changed in merged commit
            referenced_line_changed = False
            if line_to_check is not None and first_comment.get("path"):
                referenced_line_changed = is_referenced_line_changed_in_merged_commit(
                    repo=repo,
                    commit_to_review=commit_to_review,
                    file_path=first_comment.get("path"),
                    line=line_to_check,
                    merged_commit=merged_commit,
                    tokens=tokens,
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
                    marked_as_dismissed=marked_as_dismissed,
                ),
            )
            labeled_comments.append(labeled_comment)

    return labeled_comments


@cached_with_tuple_conversion
def fetch_patch_between_commits(
    repo: str, base_commit: str, head_commit: str, tokens: Optional[list[str]] = None
) -> str:
    """Extract patch between two commits."""
    github_api = GitHubAPI(tokens=tokens)
    return github_api.get_patch(repo, base_commit=base_commit, head_commit=head_commit)


@cached_with_tuple_conversion
def fetch_pr_patch(
    repo: str, pull_number: int, tokens: Optional[list[str]] = None
) -> str:
    """Extract patch for the entire PR."""
    github_api = GitHubAPI(tokens=tokens)
    return github_api.get_patch(repo, pr_number=pull_number)


@cached_with_tuple_conversion
def fetch_repo_language(repo: str, tokens: Optional[list[str]] = None) -> str:
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


@cached_with_tuple_conversion
def fetch_repo_file_content(
    repo: str, commit: str, file_path: str, tokens: Optional[list[str]] = None
) -> str:
    """Get the content of a file from a repository at a specific commit with caching."""
    github_api = GitHubAPI(tokens=tokens)
    return github_api.get_file_content(repo, commit, file_path)


def fetch_local_file_content(repo_dir: str, file_path: str) -> str:
    """Fetch the content of a file from a local repository directory."""
    try:
        full_path = os.path.join(repo_dir, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            logger.warning(f"File not found in local repo: {file_path}")
            return ""
    except Exception as e:
        logger.error(f"Failed to read local file {file_path}: {e}")
        return ""


@cached_with_tuple_conversion
def fetch_repo_files_content_by_retrieval(
    repo: str,
    commit: str,
    query: str,  # e.g., "diff_hunk" or "file_path" or "problem_statement"
    tokens: Optional[list[str]] = None,
    max_files: int = 5,
) -> dict[str, str]:
    """Get the content of specific files from a repository at a specific commit using retrieval."""
    try:
        # Clone the repository to a temporary directory if the repository doesn't exist
        if not os.path.exists(f"temp_repo/{repo.replace('/', '__')}"):
            temp_repo_dir = "temp_repo"
            os.makedirs(temp_repo_dir, exist_ok=True)
            # temp_dir = tempfile.mkdtemp(dir=temp_repo_dir)
            repo_dir = clone_repo(repo, temp_repo_dir, tokens[0] if tokens else "git")

        # Constrct the index
        python = subprocess.run("which python", shell=True, capture_output=True)
        python = python.stdout.decode("utf-8").strip()

        index_path = make_index(
            repo_dir=repo_dir,
            root_dir=temp_repo_dir,
            query="",  # Empty query to index all files
            commit=commit,
            document_encoding_func=DOCUMENT_ENCODING_FUNCTIONS[
                "file_name_and_contents"
            ],
            python=python,
            instance_id=f"{repo.replace('/', '__')}_{commit[:8]}",
        )

        # Create searcher and perform search
        # searcher = LuceneSearcher(index_path.as_posix())

        # Create search instance using retrieval_target as query
        instance = {"instance_id": f"{repo}_{commit[:8]}", "query": query}

        # Execute search
        results = search(instance, index_path)

        # Extract file contents for top results
        retrieved_files = {}
        if results and results.get("hits"):
            # specific commit
            from swe_care.utils.retrieval import ContextManager

            with ContextManager(repo_dir, commit):
                for hit in results["hits"][:max_files]:
                    file_path = hit["docid"]
                    try:
                        # Get file content from the repository at the specified commit
                        content = fetch_repo_file_content(
                            repo, commit, file_path, tokens
                        )
                        retrieved_files[file_path] = content
                        logger.debug(
                            f"Retrieved file: {file_path} (score: {hit['score']:.3f})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch content for retrieved file {file_path}: {e}"
                        )

        # Clean up temporary directory
        # shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info(
            f"Retrieved {len(retrieved_files)} files using retrieval for {repo}@{commit[:8]}"
        )
        return retrieved_files

    except Exception as e:
        logger.error(f"Failed to retrieve files using search for {repo}@{commit}: {e}")
        # Clean up on error
        # try:
        #     if 'temp_dir' in locals():
        #         shutil.rmtree(temp_dir, ignore_errors=True)
        # except:
        #     pass
        return {}

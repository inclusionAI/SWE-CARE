"""
This script uses GraphQL queries to fetch merged pull requests from a GitHub repository along with their comprehensive metadata, including:
Basic PR info, labels, commits, reviews, review comments, review threads, thread comments, and linked issues that the PR closes.

It specifically targets merged PRs ordered by creation date (newest first).

Note:
- The script filters to only include PRs that have at least 1 closing issues reference.
- The script handles pagination for all nested data.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from tqdm import tqdm

from swe_care.utils.github import GitHubAPI, MaxNodeLimitExceededError
from swe_care.utils.github_graphql import GRAPHQL_QUERIES


def fetch_all_labels(pr_id: str, initial_data: dict, github_api: GitHubAPI) -> dict:
    """Fetch all labels for a PR using pagination."""
    all_labels = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(GRAPHQL_QUERIES["labels"], variables)
        page_data = result.get("data", {}).get("node", {}).get("labels", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_labels.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_labels,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_commits(pr_id: str, initial_data: dict, github_api: GitHubAPI) -> dict:
    """Fetch all commits for a PR using pagination."""
    all_commits = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(GRAPHQL_QUERIES["commits"], variables)
        page_data = result.get("data", {}).get("node", {}).get("commits", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_commits.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_commits,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_review_comments(
    review_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all review comments for a review using pagination."""
    all_comments = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"reviewId": review_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["review_comments"], variables
        )
        page_data = result.get("data", {}).get("node", {}).get("comments", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_comments.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_comments,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_reviews(pr_id: str, initial_data: dict, github_api: GitHubAPI) -> dict:
    """Fetch all reviews for a PR using pagination, including all review comments."""
    all_reviews = []

    # Process initial reviews and fetch all their comments
    for review in initial_data.get("nodes", []):
        if review.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
            review["comments"] = fetch_all_review_comments(
                review["id"], review.get("comments", {}), github_api
            )
        all_reviews.append(review)

    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(GRAPHQL_QUERIES["reviews"], variables)
        page_data = result.get("data", {}).get("node", {}).get("reviews", {})

        if not page_data or not page_data.get("nodes"):
            break

        # Process each review and fetch all its comments
        for review in page_data.get("nodes", []):
            if review.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
                review["comments"] = fetch_all_review_comments(
                    review["id"], review.get("comments", {}), github_api
                )
            all_reviews.append(review)

        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_reviews,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_issue_labels(
    issue_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all labels for an issue using pagination."""
    all_labels = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"issueId": issue_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["issue_labels"], variables
        )
        page_data = result.get("data", {}).get("node", {}).get("labels", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_labels.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_labels,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_issue_comments(
    issue_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all comments for an issue using pagination."""
    all_comments = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"issueId": issue_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["issue_comments"], variables
        )
        page_data = result.get("data", {}).get("node", {}).get("comments", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_comments.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_comments,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_thread_comments(
    thread_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all comments for a review thread using pagination."""
    all_comments = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"threadId": thread_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["thread_comments"], variables
        )
        page_data = result.get("data", {}).get("node", {}).get("comments", {})

        if not page_data or not page_data.get("nodes"):
            break

        all_comments.extend(page_data.get("nodes", []))
        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_comments,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_review_threads(
    pr_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all review threads for a PR using pagination, including all comments for each thread."""
    all_threads = []

    # Process initial threads and fetch all their comments
    for thread in initial_data.get("nodes", []):
        if thread.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
            thread["comments"] = fetch_all_thread_comments(
                thread["id"], thread.get("comments", {}), github_api
            )
        all_threads.append(thread)

    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["review_threads"], variables
        )
        page_data = result.get("data", {}).get("node", {}).get("reviewThreads", {})

        if not page_data or not page_data.get("nodes"):
            break

        # Process each thread and fetch all its comments
        for thread in page_data.get("nodes", []):
            if thread.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
                thread["comments"] = fetch_all_thread_comments(
                    thread["id"], thread.get("comments", {}), github_api
                )
            all_threads.append(thread)

        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_threads,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_all_closing_issues(
    pr_id: str, initial_data: dict, github_api: GitHubAPI
) -> dict:
    """Fetch all closing issues references for a PR using pagination, including all labels for each issue."""
    all_issues = []

    # Process initial issues and fetch all their labels
    for issue in initial_data.get("nodes", []):
        if issue.get("labels", {}).get("pageInfo", {}).get("hasNextPage", False):
            issue["labels"] = fetch_all_issue_labels(
                issue["id"], issue.get("labels", {}), github_api
            )

        if issue.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
            issue["comments"] = fetch_all_issue_comments(
                issue["id"], issue.get("comments", {}), github_api
            )

        all_issues.append(issue)

    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(
            GRAPHQL_QUERIES["closing_issues"], variables
        )
        page_data = (
            result.get("data", {}).get("node", {}).get("closingIssuesReferences", {})
        )

        if not page_data or not page_data.get("nodes"):
            break

        # Process each issue and fetch all its labels
        for issue in page_data.get("nodes", []):
            if issue.get("labels", {}).get("pageInfo", {}).get("hasNextPage", False):
                issue["labels"] = fetch_all_issue_labels(
                    issue["id"], issue.get("labels", {}), github_api
                )
            if issue.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
                issue["comments"] = fetch_all_issue_comments(
                    issue["id"], issue.get("comments", {}), github_api
                )

            all_issues.append(issue)

        page_info = page_data.get("pageInfo", {})
        has_next_page = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return {
        "nodes": all_issues,
        "totalCount": total_count,
        "pageInfo": {"hasNextPage": False, "endCursor": None},
    }


def fetch_complete_pr_data(pr: dict, github_api: GitHubAPI) -> dict:
    """Fetch complete data for a single PR, including all nested paginated data."""
    pr_id = pr["id"]
    pr_number = pr.get("number", "unknown")
    pagination_info = []

    # Fetch all labels if there are more pages
    if pr.get("labels", {}).get("pageInfo", {}).get("hasNextPage", False):
        initial_labels = len(pr.get("labels", {}).get("nodes", []))
        pr["labels"] = fetch_all_labels(pr_id, pr.get("labels", {}), github_api)
        final_labels = len(pr["labels"].get("nodes", []))
        if final_labels > initial_labels:
            pagination_info.append(f"labels: {initial_labels}→{final_labels}")

    # Fetch all commits if there are more pages
    if pr.get("commits", {}).get("pageInfo", {}).get("hasNextPage", False):
        initial_commits = len(pr.get("commits", {}).get("nodes", []))
        pr["commits"] = fetch_all_commits(pr_id, pr.get("commits", {}), github_api)
        final_commits = len(pr["commits"].get("nodes", []))
        if final_commits > initial_commits:
            pagination_info.append(f"commits: {initial_commits}→{final_commits}")

    # Fetch all reviews and their comments if there are more pages
    if pr.get("reviews", {}).get("pageInfo", {}).get("hasNextPage", False):
        initial_reviews = len(pr.get("reviews", {}).get("nodes", []))
        pr["reviews"] = fetch_all_reviews(pr_id, pr.get("reviews", {}), github_api)
        final_reviews = len(pr["reviews"].get("nodes", []))
        if final_reviews > initial_reviews:
            pagination_info.append(f"reviews: {initial_reviews}→{final_reviews}")
    else:
        # Still need to check if individual reviews have more comments
        review_comment_pagination = []
        for review in pr.get("reviews", {}).get("nodes", []):
            if review.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
                initial_comments = len(review.get("comments", {}).get("nodes", []))
                all_comments = fetch_all_review_comments(
                    review["id"], review.get("comments", {}), github_api
                )
                review["comments"] = all_comments
                final_comments = len(all_comments.get("nodes", []))
                if final_comments > initial_comments:
                    review_comment_pagination.append(
                        f"{initial_comments}→{final_comments}"
                    )

        if review_comment_pagination:
            pagination_info.append(
                f"review comments: {', '.join(review_comment_pagination)}"
            )

    # Fetch all review threads and their comments if there are more pages
    if pr.get("reviewThreads", {}).get("pageInfo", {}).get("hasNextPage", False):
        initial_threads = len(pr.get("reviewThreads", {}).get("nodes", []))
        pr["reviewThreads"] = fetch_all_review_threads(
            pr_id, pr.get("reviewThreads", {}), github_api
        )
        final_threads = len(pr["reviewThreads"].get("nodes", []))
        if final_threads > initial_threads:
            pagination_info.append(f"review threads: {initial_threads}→{final_threads}")
    else:
        # Still need to check if individual threads have more comments
        thread_comment_pagination = []
        for thread in pr.get("reviewThreads", {}).get("nodes", []):
            if thread.get("comments", {}).get("pageInfo", {}).get("hasNextPage", False):
                initial_comments = len(thread.get("comments", {}).get("nodes", []))
                all_comments = fetch_all_thread_comments(
                    thread["id"], thread.get("comments", {}), github_api
                )
                thread["comments"] = all_comments
                final_comments = len(all_comments.get("nodes", []))
                if final_comments > initial_comments:
                    thread_comment_pagination.append(
                        f"{initial_comments}→{final_comments}"
                    )

        if thread_comment_pagination:
            pagination_info.append(
                f"thread comments: {', '.join(thread_comment_pagination)}"
            )

    # Fetch all closing issues references if there are more pages
    if (
        pr.get("closingIssuesReferences", {})
        .get("pageInfo", {})
        .get("hasNextPage", False)
    ):
        initial_issues = len(pr.get("closingIssuesReferences", {}).get("nodes", []))
        pr["closingIssuesReferences"] = fetch_all_closing_issues(
            pr_id, pr.get("closingIssuesReferences", {}), github_api
        )
        final_issues = len(pr["closingIssuesReferences"].get("nodes", []))
        if final_issues > initial_issues:
            pagination_info.append(f"closing issues: {initial_issues}→{final_issues}")
    else:
        # Still need to check if individual issues have more labels
        issue_label_pagination = []
        for issue in pr.get("closingIssuesReferences", {}).get("nodes", []):
            if issue.get("labels", {}).get("pageInfo", {}).get("hasNextPage", False):
                initial_labels = len(issue.get("labels", {}).get("nodes", []))
                all_labels = fetch_all_issue_labels(
                    issue["id"], issue.get("labels", {}), github_api
                )
                issue["labels"] = all_labels
                final_labels = len(all_labels.get("nodes", []))
                if final_labels > initial_labels:
                    issue_label_pagination.append(f"{initial_labels}→{final_labels}")

        if issue_label_pagination:
            pagination_info.append(f"issue labels: {', '.join(issue_label_pagination)}")

    # Log pagination activity if any occurred
    if pagination_info:
        logger.info(f"  PR #{pr_number}: Paginated {'; '.join(pagination_info)}")

    return pr


def get_repo_pr_data(
    repo: str,
    tokens: Optional[list[str]] = None,
    max_number: Optional[int] = None,
    after_pr_cursor: Optional[str] = None,
) -> Iterator[dict]:
    """
    Execute GraphQL query for a given repo and yield crawled PR data with complete pagination.

    This function fetches all nested data (labels, commits, reviews, review comments,
    review threads, thread comments, closing issues, and issue labels) for each PR using
    efficient multi-level pagination. It automatically handles GitHub's node limits by
    reducing page sizes when needed.

    Only yields PRs that have at least 1 closing issues reference.

    Args:
        repo: Repository in format 'owner/repo'
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch. If not provided, all PRs will be fetched.
        after_pr_cursor: Optional cursor to resume fetching after (for resuming interrupted runs)

    Yields:
        Dictionaries containing complete PR data with all nested information,
        filtered to include only PRs with closing issues references. Each closing issue
        includes all of its labels and comments with complete pagination. Each review thread
        includes all of its comments with complete pagination.
    """
    # Parse repo owner and name
    if "/" not in repo:
        raise ValueError(f"Repository must be in format 'owner/repo', got: {repo}")

    repo_owner, repo_name = repo.split("/", 1)

    # Create GitHub API instance
    github_api = GitHubAPI(tokens=tokens)

    pr_cursor = after_pr_cursor

    if max_number is None:
        logger.info(f"max_number is not provided, fetching all PRs for {repo}")
        max_number = float("inf")

    current_page_size = min(max_number, 20)  # Start with conservative page size
    total_yielded = 0

    if after_pr_cursor:
        logger.info(f"Resuming PR fetching for {repo} after cursor: {after_pr_cursor}")
    else:
        logger.info(f"Starting PR fetching for {repo} from the beginning")

    while True:
        variables = {
            "owner": repo_owner,
            "name": repo_name,
            "maxNumber": current_page_size,
            "prCursor": pr_cursor,
        }

        # Log current cursor for debugging/resuming
        if pr_cursor:
            logger.debug(f"Current PR cursor for {repo}: {pr_cursor}")

        try:
            results = github_api.execute_graphql_query(
                GRAPHQL_QUERIES["merged_pull_requests"], variables
            )
        except MaxNodeLimitExceededError as e:
            logger.warning(f"Max node limit exceeded for {repo}: {e}")
            if current_page_size > 1:
                # Reduce page size and retry
                current_page_size = max(1, current_page_size // 2)
                logger.info(
                    f"Hit node limit, reducing page size to {current_page_size} for {repo}"
                )
                continue
            else:
                logger.info(f"Cannot reduce page size further for {repo}, skipping...")
                break

        try:
            # Extract PR data
            repository_data = results.get("data", {}).get("repository")
            if not repository_data:
                raise ValueError(f"No repository data found for {repo}")

            pr_data = repository_data.get("pullRequests", {})
            prs = pr_data.get("nodes", [])

            if not prs:
                raise ValueError(f"No more PRs found for {repo}")

            # Process and yield each PR (including all nested paginated data)
            logger.info(
                f"Processing {len(prs)} PRs from page (page size: {current_page_size})..."
            )
            for i, pr in enumerate(prs, 1):
                try:
                    logger.info(
                        f"  Processing PR #{pr.get('number', 'unknown')} of repo {repo} ({i}/{len(prs)})",
                        end=" ",
                    )

                    # Filter: Only include PRs that have at least 1 closing issues reference
                    closing_issues_count = pr.get("closingIssuesReferences", {}).get(
                        "totalCount", 0
                    )
                    if closing_issues_count > 0:
                        logger.info(f"✓ (has {closing_issues_count} closing issues)")
                    else:
                        logger.info("✗ (no closing issues)")
                        continue

                    complete_pr = fetch_complete_pr_data(pr, github_api)
                    yield complete_pr
                    total_yielded += 1

                    # Check if we've reached the maximum number of PRs
                    if total_yielded >= max_number:
                        logger.info(
                            f"Reached maximum number of PRs ({max_number}) for {repo}"
                        )
                        return

                except Exception as e:
                    logger.warning(
                        f"✗ Warning: Failed to fetch complete data for PR #{pr.get('number', 'unknown')}: {e}"
                    )
                    # For error cases, still check if original PR has closing issues
                    if pr.get("closingIssuesReferences", {}).get("totalCount", 0) > 0:
                        yield pr
                        total_yielded += 1

                        # Check if we've reached the maximum number of PRs
                        if total_yielded >= max_number:
                            logger.info(
                                f"Reached maximum number of PRs ({max_number}) for {repo}"
                            )
                            return

            # Check if there are more pages
            page_info = pr_data.get("pageInfo", {})
            if not page_info.get("hasNextPage", False):
                break

            pr_cursor = page_info.get("endCursor")

        except Exception as e:
            logger.error(f"Error fetching PR data for {repo}: {e}")
            break

    logger.info(f"Collected {total_yielded} PRs with closing issues for {repo}")


def get_specific_prs_data(
    repo: str,
    pr_numbers: list[int],
    tokens: Optional[list[str]] = None,
) -> Iterator[dict]:
    """
    Fetch specific PRs by their numbers from a repository.

    Args:
        repo: Repository in format 'owner/repo'
        pr_numbers: List of PR numbers to fetch
        tokens: Optional list of GitHub tokens for API requests

    Yields:
        Dictionaries containing complete PR data for each specified PR
    """
    # Parse repo owner and name
    if "/" not in repo:
        raise ValueError(f"Repository must be in format 'owner/repo', got: {repo}")

    repo_owner, repo_name = repo.split("/", 1)

    # Create GitHub API instance
    github_api = GitHubAPI(tokens=tokens)

    logger.info(f"Fetching {len(pr_numbers)} specific PRs from {repo}")

    for pr_number in pr_numbers:
        try:
            variables = {
                "owner": repo_owner,
                "name": repo_name,
                "prNumber": pr_number,
            }

            result = github_api.execute_graphql_query(
                GRAPHQL_QUERIES["specific_pr"], variables
            )

            # Extract PR data
            repository_data = result.get("data", {}).get("repository")
            if not repository_data:
                logger.warning(f"No repository data found for {repo}")
                continue

            pr_data = repository_data.get("pullRequest")
            if not pr_data:
                logger.warning(f"PR #{pr_number} not found in {repo}")
                continue

            logger.info(f"  Processing PR #{pr_number} of repo {repo}")

            # Fetch complete data for this PR (including all nested paginated data)
            complete_pr = fetch_complete_pr_data(pr_data, github_api)
            yield complete_pr

        except Exception as e:
            logger.error(f"Error fetching PR #{pr_number} from {repo}: {e}")


def process_single_repository(
    repo_name: str,
    output_dir: Path | str,
    tokens: Optional[list[str]] = None,
    max_number: Optional[int] = None,
    specific_prs: Optional[list[int]] = None,
    after_pr_cursor: Optional[str] = None,
) -> tuple[str, int]:
    """
    Process a single repository and save PR data to file.

    Args:
        repo_name: Repository in format 'owner/repo'
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch (ignored if specific_prs is provided). If not provided, all PRs will be fetched.
        specific_prs: Optional list of specific PR numbers to fetch
        after_pr_cursor: Optional cursor to resume fetching after (for resuming interrupted runs)

    Returns:
        Tuple of (repo_name, pr_count)
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    try:
        # Create output filename
        org, repo_short = repo_name.split("/", 1)
        output_file = output_dir / f"{org}__{repo_short}_graphql_prs_data.jsonl"

        # Process PR data for this repository and write immediately
        pr_count = 0
        with output_file.open("a") as out_f:
            if specific_prs:
                # Warn if cursor is provided for specific PRs
                if after_pr_cursor:
                    logger.warning(
                        f"Ignoring --after-pr-cursor for {repo_name} when fetching specific PRs"
                    )

                # Fetch specific PRs
                for pr in get_specific_prs_data(
                    repo=repo_name,
                    pr_numbers=specific_prs,
                    tokens=tokens,
                ):
                    out_f.write(json.dumps(pr) + "\n")
                    pr_count += 1
            else:
                # Fetch all PRs with closing issues
                for pr in get_repo_pr_data(
                    repo=repo_name,
                    tokens=tokens,
                    max_number=max_number,
                    after_pr_cursor=after_pr_cursor,
                ):
                    out_f.write(json.dumps(pr) + "\n")
                    pr_count += 1

        if pr_count > 0:
            logger.info(f"Saved {pr_count} PRs to {output_file}")
        else:
            if specific_prs:
                logger.info(
                    f"No PRs found for specified numbers {specific_prs} in {repo_name}"
                )
            else:
                logger.info(f"No PRs with closing issues found for {repo_name}")

        return repo_name, pr_count

    except Exception as e:
        logger.error(f"Error processing repository {repo_name}: {e}")
        return repo_name, 0


def get_graphql_prs_data(
    repo_file: Optional[Path | str] = None,
    repo: Optional[str] = None,
    output_dir: Path | str = None,
    tokens: Optional[list[str]] = None,
    max_number: Optional[int] = None,
    specific_prs: Optional[list[int]] = None,
    jobs: int = 2,
    after_pr_cursor: Optional[str] = None,
) -> None:
    """
    Get comprehensive PR data from GitHub GraphQL API with complete pagination.

    Fetches all nested data including labels, commits, reviews, review comments,
    review threads, thread comments, closing issues, issue labels, and issue comments.
    Only fetches PRs that have at least 1 closing issues reference (when specific_prs is not provided).

    Args:
        repo_file: Path to repository file (output from get_top_repos). Each line should be a JSON
                   object with 'name' field in format 'owner/repo'. Optionally include 'pr_cursor'
                   field to resume fetching from a specific cursor for each repository.
                   Example: {"name": "owner/repo", "pr_cursor": "Y3Vyc29yOnYyOpK5MjAyNC0wNy0wOFQxNzozMDoyNFo="}
        repo: Repository in format 'owner/repo'
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch (ignored when specific_prs is provided). If not provided, all PRs will be fetched.
        specific_prs: List of specific PR numbers to fetch. If provided, only these PRs will be fetched.
        jobs: Number of concurrent jobs/threads to use (default: 2)
        after_pr_cursor: Optional cursor to resume fetching after (for resuming interrupted runs).
                        When used with repo_file, acts as fallback for repositories without pr_cursor field.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir:
        raise ValueError("output_dir is required")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate that after_pr_cursor is not used with specific_prs
    if after_pr_cursor and specific_prs:
        logger.warning("--after-pr-cursor is ignored when --specific-prs is provided")
        after_pr_cursor = None

    if repo_file:
        # Process multiple repositories from file with multi-threading
        if isinstance(repo_file, str):
            repo_file = Path(repo_file)
        if not repo_file.exists():
            raise FileNotFoundError(f"Repository file not found: {repo_file}")

        # Read all repository names and cursors first
        repo_data_list = []
        with repo_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    repo_data = json.loads(line)
                    repo_name = repo_data.get("name")

                    if not repo_name:
                        logger.warning(f"No 'name' field found in line: {line}")
                        continue

                    # Extract pr_cursor if available, use global cursor as fallback
                    repo_cursor = repo_data.get("pr_cursor")
                    if after_pr_cursor and repo_cursor:
                        logger.warning(
                            f"Both --after-pr-cursor and pr_cursor in file found for {repo_name}, using file cursor: {repo_cursor}"
                        )
                    elif after_pr_cursor and not repo_cursor:
                        repo_cursor = after_pr_cursor
                        logger.info(
                            f"Using global --after-pr-cursor for {repo_name}: {repo_cursor}"
                        )

                    repo_data_list.append({"name": repo_name, "cursor": repo_cursor})

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON line: {line}, error: {e}")

        if not repo_data_list:
            logger.warning("No valid repositories found in file")
            return

        if specific_prs:
            logger.info(
                f"Processing {len(repo_data_list)} repositories with {jobs} concurrent jobs (fetching specific PRs: {specific_prs})..."
            )
        else:
            cursor_info = sum(1 for repo in repo_data_list if repo["cursor"])
            if cursor_info > 0:
                logger.info(
                    f"Processing {len(repo_data_list)} repositories with {jobs} concurrent jobs ({cursor_info} with resume cursors)..."
                )
            else:
                logger.info(
                    f"Processing {len(repo_data_list)} repositories with {jobs} concurrent jobs..."
                )

        # Process repositories using ThreadPoolExecutor
        total_prs = 0
        successful_repos = 0

        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all jobs
            future_to_repo = {
                executor.submit(
                    process_single_repository,
                    repo_data["name"],
                    output_dir,
                    tokens,
                    max_number,
                    specific_prs,
                    repo_data["cursor"],
                ): repo_data["name"]
                for repo_data in repo_data_list
            }

            # Process completed jobs with progress bar
            with tqdm(
                total=len(repo_data_list), desc="Processing repositories"
            ) as pbar:
                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        processed_repo, pr_count = future.result()
                        total_prs += pr_count
                        if pr_count > 0:
                            successful_repos += 1
                        pbar.set_postfix(
                            {
                                "PRs": total_prs,
                                "Success": f"{successful_repos}/{len(repo_data_list)}",
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Repository {repo_name} generated an exception: {e}"
                        )
                    finally:
                        pbar.update(1)

        logger.info(
            f"Completed processing. Total PRs collected: {total_prs} from {successful_repos}/{len(repo_data_list)} repositories"
        )

    elif repo:
        if specific_prs:
            logger.info(
                f"Processing single repository: {repo} (fetching specific PRs: {specific_prs}), no threading needed"
            )
        else:
            logger.info(f"Processing single repository: {repo}, no threading needed")

        repo_name, pr_count = process_single_repository(
            repo, output_dir, tokens, max_number, specific_prs, after_pr_cursor
        )

    else:
        raise ValueError("Either repo_file or repo must be specified")

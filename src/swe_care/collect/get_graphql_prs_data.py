import json
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from swe_care.utils.github import GitHubAPI, MaxNodeLimitExceededError

# GitHub GraphQL API endpoint
GRAPHQL_ENDPOINT = "https://api.github.com/graphql"

# --- GraphQL Queries ---
# Main query for fetching PRs with first page of nested data
"""
This GraphQL query fetches merged pull requests from a GitHub repository along with their comprehensive metadata, including:
Basic PR info, labels, commits, reviews, review comments, review threads, thread comments, and linked issues that the PR closes.

It specifically targets merged PRs ordered by creation date (newest first).
Note: The script filters to only include PRs that have at least 1 closing issues reference.
"""

MAIN_GRAPHQL_QUERY = """
query GetMergedPullRequests($owner: String!, $name: String!, $prCursor: String, $maxNumber: Int!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    pullRequests(states: [MERGED], first: $maxNumber, after: $prCursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        id
        title
        body
        number
        url
        author {
          login
        }
        createdAt
        mergedAt
        mergedBy {
          login
        }
        baseRefOid
        baseRefName
        headRefOid
        headRefName
        changedFiles

        labels(first: 10) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            name
          }
        }

        commits(first: 10) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            commit {
              oid
              message
              changedFilesIfAvailable
              authoredDate
              author {
                user {
                  login
                }
              }
              committedDate
              pushedDate
              committer {
                user {
                  login
                }
              }
              parents(first: 2) {
                nodes {
                  oid
                }
              }
            }
          }
        }

        reviews(first: 10) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            id
            author {
              login
            }
            state
            body
            submittedAt
            updatedAt
            commit {
              oid
            }
            comments(first: 10) {
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
                author {
                  login
                }
                body
                createdAt
                updatedAt
                path
                diffHunk
                line
                startLine
                originalLine
                originalStartLine
                replyTo {
                    id
                }
                originalCommit {
                    oid
                }
              }
            }
          }
        }

        reviewThreads(first: 10) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            id
            isResolved
            isOutdated
            isCollapsed
            path
            startLine
            originalLine
            diffSide
            startDiffSide
            resolvedBy {
              login
            }
            comments(first: 10){
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
              }
            }
          }
        }

        closingIssuesReferences(first: 10) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            id
            number
            url
            title
            body
            state
            labels(first: 10) {
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                name
              }
            }
            comments(first: 10) {
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
                author {
                  login
                }
                body
                createdAt
                updatedAt
              }
            }
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of labels
LABELS_QUERY = """
query GetLabels($prId: ID!, $cursor: String) {
  node(id: $prId) {
    ... on PullRequest {
      labels(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          name
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of commits
COMMITS_QUERY = """
query GetCommits($prId: ID!, $cursor: String) {
  node(id: $prId) {
    ... on PullRequest {
      commits(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          commit {
            oid
            message
            changedFilesIfAvailable
            authoredDate
            author {
              user {
                login
              }
            }
            committedDate
            committer {
              user {
                login
              }
            }
            parents(first: 2) {
              nodes {
                oid
              }
            }
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of reviews
REVIEWS_QUERY = """
query GetReviews($prId: ID!, $cursor: String) {
  node(id: $prId) {
    ... on PullRequest {
      reviews(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          author {
            login
          }
          state
          body
          submittedAt
          updatedAt
          commit {
            oid
          }
          comments(first: 100) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              id
              author {
                login
              }
              body
              createdAt
              updatedAt
              path
              diffHunk
              line
              startLine
              originalLine
              originalStartLine
              replyTo {
                id
              }
              originalCommit {
                oid
              }
            }
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of review comments
REVIEW_COMMENTS_QUERY = """
query GetReviewComments($reviewId: ID!, $cursor: String) {
  node(id: $reviewId) {
    ... on PullRequestReview {
      comments(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          author {
            login
          }
          body
          createdAt
          updatedAt
          path
          diffHunk
          line
          startLine
          originalLine
          originalStartLine
          replyTo {
            id
          }
          originalCommit {
            oid
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of closing issues references
CLOSING_ISSUES_QUERY = """
query GetClosingIssues($prId: ID!, $cursor: String) {
  node(id: $prId) {
    ... on PullRequest {
      closingIssuesReferences(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          number
          url
          title
          body
          state
          labels(first: 10) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              name
            }
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of issue labels
ISSUE_LABELS_QUERY = """
query GetIssueLabels($issueId: ID!, $cursor: String) {
  node(id: $issueId) {
    ... on Issue {
      labels(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          name
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of issue comments
ISSUE_COMMENTS_QUERY = """
query GetIssueComments($issueId: ID!, $cursor: String) {
  node(id: $issueId) {
    ... on Issue {
      comments(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          author {
            login
          }
          body
          createdAt
          updatedAt
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of review threads
REVIEW_THREADS_QUERY = """
query GetReviewThreads($prId: ID!, $cursor: String) {
  node(id: $prId) {
    ... on PullRequest {
      reviewThreads(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          isResolved
          isOutdated
          isCollapsed
          path
          startLine
          originalLine
          diffSide
          startDiffSide
          resolvedBy {
            login
          }
          comments(first: 10) {
            totalCount
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              id
            }
          }
        }
      }
    }
  }
}
"""

# Query for fetching additional pages of thread comments
THREAD_COMMENTS_QUERY = """
query GetThreadComments($threadId: ID!, $cursor: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      comments(first: 100, after: $cursor) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
        }
      }
    }
  }
}
"""


def fetch_all_labels(pr_id: str, initial_data: dict, github_api: GitHubAPI) -> dict:
    """Fetch all labels for a PR using pagination."""
    all_labels = list(initial_data.get("nodes", []))
    cursor = initial_data.get("pageInfo", {}).get("endCursor")
    total_count = initial_data.get("totalCount", 0)
    has_next_page = initial_data.get("pageInfo", {}).get("hasNextPage", False)

    while has_next_page and cursor:
        variables = {"prId": pr_id, "cursor": cursor}
        result = github_api.execute_graphql_query(LABELS_QUERY, variables)
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
        result = github_api.execute_graphql_query(COMMITS_QUERY, variables)
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
        result = github_api.execute_graphql_query(REVIEW_COMMENTS_QUERY, variables)
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
        result = github_api.execute_graphql_query(REVIEWS_QUERY, variables)
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
        result = github_api.execute_graphql_query(ISSUE_LABELS_QUERY, variables)
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
        result = github_api.execute_graphql_query(ISSUE_COMMENTS_QUERY, variables)
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
        result = github_api.execute_graphql_query(THREAD_COMMENTS_QUERY, variables)
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
        result = github_api.execute_graphql_query(REVIEW_THREADS_QUERY, variables)
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
        result = github_api.execute_graphql_query(CLOSING_ISSUES_QUERY, variables)
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
    max_number: int = 10,
) -> list[dict]:
    """
    Execute GraphQL query for a given repo and return crawled PR data with complete pagination.

    This function fetches all nested data (labels, commits, reviews, review comments,
    review threads, thread comments, closing issues, and issue labels) for each PR using
    efficient multi-level pagination. It automatically handles GitHub's node limits by
    reducing page sizes when needed.

    Only returns PRs that have at least 1 closing issues reference.

    Args:
        repo: Repository in format 'owner/repo'
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch

    Returns:
        List of dictionaries containing complete PR data with all nested information,
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

    all_prs = []
    pr_cursor = None
    current_page_size = min(max_number, 20)  # Start with conservative page size

    while True:
        variables = {
            "owner": repo_owner,
            "name": repo_name,
            "maxNumber": current_page_size,
            "prCursor": pr_cursor,
        }

        try:
            results = github_api.execute_graphql_query(MAIN_GRAPHQL_QUERY, variables)
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

            # Fetch complete data for each PR (including all nested paginated data)
            complete_prs = []
            logger.info(
                f"Fetching complete data for {len(prs)} PRs from page (page size: {current_page_size})..."
            )
            for i, pr in enumerate(prs, 1):
                try:
                    logger.info(
                        f"  Processing PR #{pr.get('number', 'unknown')} ({i}/{len(prs)})",
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
                    complete_prs.append(complete_pr)
                except Exception as e:
                    logger.warning(
                        f"✗ Warning: Failed to fetch complete data for PR #{pr.get('number', 'unknown')}: {e}"
                    )
                    # For error cases, still check if original PR has closing issues
                    if pr.get("closingIssuesReferences", {}).get("totalCount", 0) > 0:
                        complete_prs.append(pr)

            all_prs.extend(complete_prs)

            # Log filtering statistics
            if complete_prs:
                logger.info(
                    f"  → Included {len(complete_prs)} PRs with closing issues from this page"
                )
            else:
                logger.info("  → No PRs with closing issues found on this page")

            # Check if we've reached the maximum number of PRs
            if len(all_prs) >= max_number:
                # all_prs = all_prs[:max_number]  # Trim to exact max_number
                break

            # Check if there are more pages
            page_info = pr_data.get("pageInfo", {})
            if not page_info.get("hasNextPage", False):
                break

            pr_cursor = page_info.get("endCursor")

        except Exception as e:
            logger.error(f"Error fetching PR data for {repo}: {e}")
            break

    logger.info(f"Collected {len(all_prs)} PRs with closing issues for {repo}")
    return all_prs


def get_graphql_prs_data(
    repo_file: Optional[Path] = None,
    repo: Optional[str] = None,
    output_dir: Path = None,
    tokens: Optional[list[str]] = None,
    max_number: int = 10,
) -> None:
    """
    Get comprehensive PR data from GitHub GraphQL API with complete pagination.

    Fetches all nested data including labels, commits, reviews, review comments,
    review threads, thread comments, closing issues, issue labels, and issue comments.
    Only fetches PRs that have at least 1 closing issues reference.

    Args:
        repo_file: Path to repository file (output from get_top_repos)
        repo: Repository in format 'owner/repo'
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch
    """
    if not output_dir:
        raise ValueError("output_dir is required")

    output_dir.mkdir(parents=True, exist_ok=True)

    if repo_file:
        # Process multiple repositories from file
        if not repo_file.exists():
            raise FileNotFoundError(f"Repository file not found: {repo_file}")

        # Count total lines for progress bar
        with repo_file.open("r") as f:
            total_repos = sum(1 for _ in f)

        with repo_file.open("r") as f:
            for line in tqdm(f, total=total_repos, desc="Processing repositories"):
                line = line.strip()
                if not line:
                    continue

                try:
                    repo_data = json.loads(line)
                    repo_name = repo_data.get("name")

                    if not repo_name:
                        logger.warning(f"No 'name' field found in line: {line}")
                        continue

                    # Get PR data for this repository
                    pr_data = get_repo_pr_data(
                        repo=repo_name,
                        tokens=tokens,
                        max_number=max_number,
                    )

                    if pr_data:
                        # Create output filename
                        org, repo_short = repo_name.split("/", 1)
                        output_file = (
                            output_dir / f"{org}__{repo_short}_graphql_prs_data.jsonl"
                        )

                        # Save PR data to file
                        with output_file.open("w") as out_f:
                            for pr in pr_data:
                                out_f.write(json.dumps(pr) + "\n")

                        logger.info(f"Saved {len(pr_data)} PRs to {output_file}")

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON line: {line}, error: {e}")
                except Exception as e:
                    logger.error(
                        f"Error processing repository {repo_name if 'repo_name' in locals() else 'unknown'}: {e}"
                    )

    elif repo:
        # Process single repository
        pr_data = get_repo_pr_data(
            repo=repo,
            tokens=tokens,
            max_number=max_number,
        )

        if pr_data:
            # Create output filename
            org, repo_short = repo.split("/", 1)
            output_file = output_dir / f"{org}__{repo_short}_graphql_prs_data.jsonl"

            # Save PR data to file
            with output_file.open("w") as out_f:
                for pr in pr_data:
                    out_f.write(json.dumps(pr) + "\n")

            logger.info(f"Saved {len(pr_data)} PRs to {output_file}")

    else:
        raise ValueError("Either repo_file or repo must be specified")

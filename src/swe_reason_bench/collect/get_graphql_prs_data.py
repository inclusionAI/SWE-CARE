import json
import random
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# GitHub GraphQL API endpoint
GRAPHQL_ENDPOINT = "https://api.github.com/graphql"

# --- GraphQL Query (as defined above) ---
GRAPHQL_QUERY = """
query GetMergedPullRequests($owner: String!, $name: String!, $prCursor: String, $commitsCursor: String, $reviewsCursor: String, $reviewCommentsCursor: String, $issuesCursor: String, $labelsCursor: String, $maxNumber: Int!) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    pullRequests(states: [MERGED], first: $maxNumber, after: $prCursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        title
        number
        url
        author {
          login
        }
        mergedAt
        mergedBy {
          login
        }
        baseRefName
        headRefName
        additions
        deletions
        changedFiles

        labels(first: 10, after: $labelsCursor) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            name
            color
          }
        }

        commits(first: 10, after: $commitsCursor) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            commit {
              oid
              messageHeadline
              messageBody
              authoredDate
              author {
                name
                email
                user {
                  login
                }
              }
              committedDate
              committer {
                name
                email
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

        reviews(first: 50, after: $reviewsCursor) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            author {
              login
            }
            state
            body
            submittedAt
            comments(first: 50, after: $reviewCommentsCursor) {
              totalCount
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                author {
                  login
                }
                body
                createdAt
                path
                diffHunk
                line
                startLine
              }
            }
          }
        }

        closingIssuesReferences(first: 5, after: $issuesCursor) {
          totalCount
          pageInfo {
            hasNextPage
            endCursor
          }
          nodes {
            number
            title
            url
            state
          }
        }
      }
    }
  }
}
"""


def get_repo_pr_data(
    repo: str,
    tokens: Optional[list[str]] = None,
    max_number: int = 10,
) -> list[dict]:
    """
    Execute GraphQL query for a given repo and return crawled PR data with pagination.

    Args:
        repo: Repository in format 'owner/repo'
        tokens: Optional list of GitHub tokens for API requests
        max_number: Maximum number of PRs to fetch

    Returns:
        List of dictionaries containing PR data
    """
    # Parse repo owner and name
    if "/" not in repo:
        raise ValueError(f"Repository must be in format 'owner/repo', got: {repo}")

    repo_owner, repo_name = repo.split("/", 1)

    # Setup headers
    headers = {
        "Content-Type": "application/json",
    }

    if tokens:
        # Use a random token if provided
        token = random.choice(tokens)
        headers["Authorization"] = f"bearer {token}"

    all_prs = []
    pr_cursor = None
    retry_count = 0
    max_retries = 3

    while True:
        variables = {
            "owner": repo_owner,
            "name": repo_name,
            # GitHub GraphQL API max is 100 per page
            "maxNumber": min(max_number, 100),
            "prCursor": pr_cursor,
            "commitsCursor": None,
            "reviewsCursor": None,
            "reviewCommentsCursor": None,
            "issuesCursor": None,
            "labelsCursor": None,
        }

        payload = {
            "query": GRAPHQL_QUERY,
            "variables": variables,
        }

        try:
            response = requests.post(
                GRAPHQL_ENDPOINT, headers=headers, data=json.dumps(payload)
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            results = response.json()

            # Check for GraphQL errors
            if "errors" in results:
                raise ValueError(f"GraphQL errors for {repo}: {results['errors']}")

            # Extract PR data
            repository_data = results.get("data", {}).get("repository")
            if not repository_data:
                raise ValueError(f"No repository data found for {repo}")

            pr_data = repository_data.get("pullRequests", {})
            prs = pr_data.get("nodes", [])

            if not prs:
                raise ValueError(f"No more PRs found for {repo}")

            all_prs.extend(prs)

            # Check if we've reached the maximum number of PRs
            if len(all_prs) >= max_number:
                all_prs = all_prs[:max_number]  # Trim to exact max_number
                break

            # Reset retry count on successful request
            retry_count = 0

            # Check if there are more pages
            page_info = pr_data.get("pageInfo", {})
            if not page_info.get("hasNextPage", False):
                break

            pr_cursor = page_info.get("endCursor")

            # Check rate limiting
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                if remaining < 10:  # If less than 10 requests remaining
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    current_time = int(time.time())
                    wait_time = max(0, reset_time - current_time)
                    if wait_time > 0:
                        print(f"Rate limit approaching. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)

            # Small delay to be respectful to the API
            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            retry_count += 1
            print(
                f"Error fetching PR data for {repo} (attempt {retry_count}/{max_retries}): {e}"
            )

            if retry_count >= max_retries:
                print(f"Max retries ({max_retries}) reached for {repo}. Breaking loop.")
                break

            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 403:
                    print("Rate limit exceeded. Try using GitHub tokens or waiting.")

            # Wait before retrying
            time.sleep(2**retry_count)  # Exponential backoff

    print(f"Collected {len(all_prs)} PRs for {repo}")
    return all_prs


def get_graphql_prs_data(
    repo_file: Optional[Path] = None,
    repo: Optional[str] = None,
    output_dir: Path = None,
    tokens: Optional[list[str]] = None,
    top_n: int = 10,
    max_number: int = 10,
) -> None:
    """
    Get PR data from GitHub GraphQL API.

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
                        print(f"No 'name' field found in line: {line}")
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
                            output_dir / f"{org}__{repo_short}_prs_with_issues.jsonl"
                        )

                        # Save PR data to file
                        with output_file.open("w") as out_f:
                            for pr in pr_data:
                                out_f.write(json.dumps(pr) + "\n")

                        print(f"Saved {len(pr_data)} PRs to {output_file}")

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {line}, error: {e}")
                except Exception as e:
                    print(
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

            print(f"Saved {len(pr_data)} PRs to {output_file}")

    else:
        raise ValueError("Either repo_file or repo must be specified")

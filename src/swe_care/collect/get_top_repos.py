"""
Fetch top repositories for a given language and save to JSONL file.
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from swe_care.utils.github import GitHubAPI


def get_top_repos(
    language: str,
    top_n: int,
    output_dir: Path | str,
    tokens: Optional[list[str]] = None,
) -> None:
    """
    Fetch top repositories for a given language and save to JSONL file.

    Args:
        language: Programming language to search for
        top_n: Number of top repositories to fetch
        output_dir: Directory to save the output file
        tokens: Optional list of GitHub tokens for API requests
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"repos_top_{top_n}_{language}.jsonl"

    # Create GitHub API instance
    github_api = GitHubAPI(tokens=tokens)

    repos_collected = 0
    page = 1

    with open(output_file, "w") as f:
        while repos_collected < top_n:
            remaining_results = top_n - repos_collected
            per_page = min(100, remaining_results)  # GitHub API max is 100 per page

            params = {
                "q": f"language:{language}",
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }

            logger.info(f"Fetching page {page} with {per_page} results...")

            try:
                response = github_api.call_api("search/repositories", params=params)
                data = response.json()
                items = data.get("items", [])

                if not items:
                    logger.info(
                        f"No more repositories found. Collected {repos_collected} repositories."
                    )
                    break

                for repo in items:
                    if repos_collected >= top_n:
                        break

                    repo_data = {
                        "name": repo["full_name"],
                        "stars": repo["stargazers_count"],
                        "url": repo["html_url"],
                        "description": repo.get("description", ""),
                        "owner": repo["owner"]["login"],
                        "language": repo.get("language", ""),
                    }

                    f.write(json.dumps(repo_data) + "\n")
                    repos_collected += 1

                logger.info(f"Collected {repos_collected}/{top_n} repositories")

                page += 1

            except Exception as e:
                logger.error(f"Error fetching repositories: {e}")
                break

    logger.success(
        f"Successfully saved {repos_collected} repositories to {output_file}"
    )

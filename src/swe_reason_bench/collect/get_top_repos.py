import json
import random
import time
from pathlib import Path
from typing import Optional

import requests


def get_top_repos(
    language: str, top_n: int, output_dir: Path, tokens: Optional[list[str]] = None
) -> None:
    """
    Fetch top repositories for a given language and save to JSONL file.

    Args:
        language: Programming language to search for
        top_n: Number of top repositories to fetch
        output_dir: Directory to save the output file
        tokens: Optional list of GitHub tokens for API requests
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"repos_top_{top_n}_{language}.jsonl"

    # Setup headers for API requests
    headers = {
        "Accept": "application/vnd.github+json",
    }

    if tokens:
        # Use a random token if provided
        token = random.choice(tokens)
        headers["Authorization"] = f"token {token}"

    base_url = "https://api.github.com/search/repositories"

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

            print(f"Fetching page {page} with {per_page} results...")

            try:
                response = requests.get(base_url, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                items = data.get("items", [])

                if not items:
                    print(
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

                print(f"Collected {repos_collected}/{top_n} repositories")

                # Check rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 10:  # If less than 10 requests remaining
                        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                        current_time = int(time.time())
                        wait_time = max(0, reset_time - current_time)
                        if wait_time > 0:
                            print(
                                f"Rate limit approaching. Waiting {wait_time} seconds..."
                            )
                            time.sleep(wait_time)

                page += 1

                # Small delay to be respectful to the API
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching repositories: {e}")
                if hasattr(e, "response") and e.response is not None:
                    if e.response.status_code == 403:
                        print(
                            "Rate limit exceeded. Try using GitHub tokens or waiting."
                        )
                break

    print(f"Successfully saved {repos_collected} repositories to {output_file}")

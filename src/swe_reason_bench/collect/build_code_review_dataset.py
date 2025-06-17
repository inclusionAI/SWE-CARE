from pathlib import Path
from typing import Optional


def build_code_review_dataset(
    repo_file: Optional[Path] = None,
    repo: Optional[str] = None,
    output_dir: Path = None,
    tokens: Optional[list[str]] = None,
) -> None:
    """
    Build code review task dataset.

    Args:
        repo_file: Path to repository file (output from get_top_repos)
        repo: Repository in format 'owner/repo'
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
    """
    # TODO
    print("build_code_review_dataset function called")
    print(f"repo_file: {repo_file}")
    print(f"repo: {repo}")
    print(f"output_dir: {output_dir}")
    print(f"tokens: {tokens}")
    pass

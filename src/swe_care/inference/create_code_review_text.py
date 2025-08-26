"""
Generate text datasets from SWE-CARE with specified prompts and context sources.

This module creates datasets in the format required for SWE-CARE evaluation by processing
the original dataset and applying different file source strategies (oracle, bm25, or all).
"""

import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

from loguru import logger
from tqdm import tqdm

from swe_care.schema.dataset import CodeReviewTaskInstance
from swe_care.schema.inference import CodeReviewInferenceInstance
from swe_care.utils.bm25_retrieval import (
    DOCUMENT_ENCODING_FUNCTIONS,
    build_documents,
    clone_repo,
)
from swe_care.utils.extract_prs_data import (
    fetch_repo_file_content,
    fetch_repo_files_content_by_retrieval,
)
from swe_care.utils.load import load_code_review_dataset
from swe_care.utils.patch import get_changed_file_paths
from swe_care.utils.template import render_template


def create_code_review_text(
    dataset_file: Path | str,
    output_dir: Path | str,
    file_source: Literal["none", "oracle", "bm25", "all"],
    k: int | None = None,
    retrieval_output_dir: Path | None = None,
    tokens: list[str] | None = None,
    jobs: int = 2,
) -> None:
    """
    Generate text datasets from SWE-CARE with specified prompts and context sources.

    Args:
        dataset_file: Path to the input SWE-CARE dataset
        output_dir: Directory to save the generated text dataset
        file_source: Source strategy for files - 'none', 'oracle', 'bm25', or 'all'
        k: Maximum number of files to use for retrieval
        retrieval_output_dir: Output directory for retrieval operations (required for bm25 and all file_source)
        tokens: GitHub API tokens (optional)
        jobs: Number of parallel jobs for multithreaded processing (default: 2)
    """
    logger.info(
        f"Starting create_code_review_text with file_source={file_source}, jobs={jobs}"
    )

    # Validate arguments
    if file_source in ["bm25", "all"]:
        if retrieval_output_dir is None:
            raise ValueError(
                f"--retrieval-output-dir is required when --file-source is '{file_source}'"
            )
        if k is None:
            raise ValueError(f"--k is required when --file-source is '{file_source}'")

    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_file}")

    # Load the dataset
    instances = load_code_review_dataset(dataset_file)

    # Process dataset and generate text
    success_count = 0
    failed_count = 0

    # Create output file and prepare for continuous writing
    if k is not None and file_source in ["bm25", "all"]:
        output_file = output_dir / f"{dataset_file.stem}__{file_source}__k{k}.jsonl"
    else:
        output_file = output_dir / f"{dataset_file.stem}__{file_source}.jsonl"
    logger.info(f"Will save processed instances to {output_file}")

    # File lock for thread-safe writing
    file_lock = Lock()

    # Choose executor based on file_source
    # Use ProcessPoolExecutor for bm25 and all to enable better parallelism
    # with file system operations and git worktrees
    if file_source in ["bm25", "all"]:
        executor_class = ProcessPoolExecutor
        logger.info(f"Using ProcessPoolExecutor with {jobs} workers for {file_source}")
    else:
        executor_class = ThreadPoolExecutor
        logger.info(f"Using ThreadPoolExecutor with {jobs} threads")

    with open(output_file, "w") as f, executor_class(max_workers=jobs) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(
                create_code_review_text_instance,
                instance,
                file_source,
                k,
                retrieval_output_dir,
                tokens,
            ): instance
            for instance in instances
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(instances), desc="Processing instances") as pbar:
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]

                try:
                    prediction = future.result()
                    with file_lock:
                        f.write(prediction.to_json() + "\n")
                    success_count += 1

                except Exception as e:
                    failed_count += 1
                    logger.error(
                        f"Exception processing instance {instance.instance_id}: {e}"
                    )

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success": success_count,
                        "failed": failed_count,
                    }
                )

    logger.info(f"Successfully generated {success_count} text instances")
    logger.info(f"Output saved to {output_file}")


def create_code_review_text_instance(
    instance: CodeReviewTaskInstance,
    file_source: Literal["none", "oracle", "bm25", "all"],
    k: int | None = None,
    retrieval_output_dir: Path | None = None,
    tokens: list[str] | None = None,
) -> CodeReviewInferenceInstance:
    """
    Process a single instance from the dataset.

    Args:
        instance: Single instance from the dataset
        file_source: Source strategy for files
        k: Maximum number of files to use
        retrieval_output_dir: Output directory for retrieval operations (if applicable)
        tokens: GitHub API tokens (optional)

    Returns:
        Processed instance with text content
    """
    # Get files based on the source strategy
    if file_source == "none":
        files = {}  # No files for "none" strategy
    elif file_source == "oracle":
        files = get_oracle_files(instance, tokens)
    elif file_source == "bm25":
        files = get_bm25_files(instance, retrieval_output_dir, k, tokens)
    elif file_source == "all":
        files = get_all_files(instance, retrieval_output_dir, k, tokens)
    else:
        raise ValueError(f"Unknown file_source: {file_source}")

    # Generate context text from files
    context_text = generate_context_text(instance, files)

    return CodeReviewInferenceInstance(
        **instance.to_dict(),
        text=context_text,
    )


def get_oracle_files(
    instance: CodeReviewTaskInstance,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get file path and file content using oracle strategy (ground truth files).
    Ground truth files are the changed files in `diff(base_commit, commit_to_review) U diff(base_commit, merged_commit)`.
    """
    changed_files = {}

    repo = instance.repo
    base_commit = instance.base_commit
    commit_to_review = instance.commit_to_review.head_commit
    merged_commit = instance.merged_commit

    logger.debug(f"Getting changed file paths from {base_commit} to {commit_to_review}")
    review_commit_changed_file_paths = get_changed_file_paths(
        instance.commit_to_review.patch_to_review
    )
    logger.debug(f"Changed file paths: {review_commit_changed_file_paths}")

    logger.debug(f"Getting changed file paths from {base_commit} to {merged_commit}")
    merged_commit_changed_file_paths = get_changed_file_paths(instance.merged_patch)
    logger.debug(f"Changed file paths: {merged_commit_changed_file_paths}")

    changed_file_paths = set(review_commit_changed_file_paths) | set(
        merged_commit_changed_file_paths
    )

    # Fetch file contents
    for file_path in changed_file_paths:
        try:
            logger.debug(f"Fetching content for {file_path}")
            content = fetch_repo_file_content(repo, base_commit, file_path, tokens)
            changed_files[file_path] = content
        except Exception as e:
            logger.warning(f"Failed to fetch content for {file_path}: {e}")
            changed_files[file_path] = ""

    # Filter out files without content and return only the files we fetched
    result = {
        path: content for path, content in changed_files.items() if content is not None
    }

    logger.info(
        f"Retrieved {len(result)} oracle files for instance {instance.instance_id}"
    )
    return result


def get_bm25_files(
    instance: CodeReviewTaskInstance,
    retrieval_output_dir: Path | None,
    k: int,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get files using BM25 retrieval based on problem statement.

    Args:
        instance: The code review task instance
        retrieval_output_dir: Output directory for retrieval operations
        k: Maximum number of files to retrieve (default: 5)
        tokens: GitHub API tokens (optional)

    Returns:
        Dictionary mapping file paths to file contents
    """

    if retrieval_output_dir is None:
        raise ValueError("retrieval_output_dir is required for BM25 file source")

    # Use problem statement as query for BM25 retrieval
    query = instance.problem_statement

    if not query.strip():
        logger.warning(
            f"Empty problem statement for instance {instance.instance_id}, using PR title and body"
        )
        # Fallback to PR title and body if problem statement is empty
        query = f"{instance.title}\n{instance.body}".strip()

    # Retrieve files using BM25
    files = fetch_repo_files_content_by_retrieval(
        repo=instance.repo,
        commit=instance.base_commit,
        query=query,
        retrieval_output_dir=retrieval_output_dir,
        tokens=tokens,
        max_files=k,
    )

    logger.info(
        f"Retrieved {len(files)} files for instance {instance.instance_id} using BM25"
    )
    return files


def get_all_files(
    instance: CodeReviewTaskInstance,
    retrieval_output_dir: Path | None,
    k: int,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get all available files from the repository up to k limit.

    This retrieves all files from the repository at the base commit using
    a temporary git worktree for thread-safety and proper cleanup.

    Args:
        instance: The code review task instance
        retrieval_output_dir: Output directory for git operations
        k: Maximum number of files to retrieve (if None, retrieves all files)
        tokens: GitHub API tokens (optional)

    Returns:
        Dictionary mapping file paths to file contents
    """

    if retrieval_output_dir is None:
        raise ValueError("retrieval_output_dir is required for 'all' file source")

    all_files = {}

    try:
        # Setup repo path
        repo_path = Path(
            f"{retrieval_output_dir}/repos/{instance.repo.replace('/', '__')}"
        )

        # Clone the repository if it doesn't exist
        if not repo_path.exists():
            Path(retrieval_output_dir).mkdir(parents=True, exist_ok=True)
            repo_dir = clone_repo(
                instance.repo,
                retrieval_output_dir,
                random.choice(tokens) if tokens else None,
            )
        else:
            repo_dir = str(repo_path)

        all_files = build_documents(
            repo_dir,
            instance.base_commit,
            DOCUMENT_ENCODING_FUNCTIONS["contents_only"],
            include_readmes=True,
        )

        # Limit the number of files
        if len(all_files) > k:
            # Convert to list, slice to k items, then convert back to dict
            all_files_items = list(all_files.items())[:k]
            all_files = dict(all_files_items)

        logger.info(
            f"Retrieved {len(all_files)} files for instance {instance.instance_id} using 'all' file source"
        )

    except Exception as e:
        logger.error(
            f"Failed to retrieve all files for {instance.repo}@{instance.base_commit}: {e}"
        )
        # Return empty dict on failure
        return {}

    return all_files


def generate_context_text(
    instance: CodeReviewTaskInstance,
    files: dict[str, str],
    add_line_numbers: bool = True,
) -> str:
    """
    Generate context text from the selected files.

    Args:
        instance: The code review task instance
        files: Dictionary mapping file paths to file contents
        add_line_numbers: Whether to add line numbers to the start of each line (default: True)

    Returns:
        Generated context text
    """
    # Render the template with the provided context
    return render_template(
        "code_review_text_prompt.j2",
        problem_statement=instance.problem_statement,
        files=files,
        patch=instance.commit_to_review.patch_to_review,
        add_line_numbers=add_line_numbers,
    )

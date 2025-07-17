"""
Generate text datasets from SWE-CARE with specified prompts and context sources.

This module creates datasets in the format required for SWE-CARE evaluation by processing
the original dataset and applying different file source strategies (oracle, bm25, or all).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Literal

from loguru import logger
from tqdm import tqdm

from swe_care.schema.dataset import CodeReviewTaskInstance
from swe_care.schema.inference import CodeReviewInferenceInstance
from swe_care.utils.extract_prs_data import fetch_repo_file_content
from swe_care.utils.load import load_code_review_dataset
from swe_care.utils.patch import get_changed_file_paths

PROMPT_PREFIX = """You will be provided with 1) a partial code base, 2) an issue statement explaining a problem to resolve, and 3) a patch addressing the issue.
<issue>
{problem_statement}
</issue>
"""


REVIEW_PROMPT = """I need you to comprehensively review the above patch with respect to the given issue from multiple dimensions such as functional implementation, code quality, and defects. Generate a report in the following format.
<review>
## Function
Briefly describe the main purpose and implemented functionality of this patch. Specifically, does this patch address the issue?

## Complexity
Is the patch more complex than it should be? Check it at every level - lines, functions, classes.

## Style
Does the patch follow the programming conventions of the original code? Does it pick good names for newly introduced variables, functions, classes, and files?

## Documentation
Does the patch provide clear and necessary comments? If the patch changes how users build, test, interact with, or release code, does it also update associated documentation?

## Defects
If there are any defects in the patch, point out their locations (file path and line number) and provide improvement suggestions in the following format:
<defect>
file_path: file path of defect 1 in the patch
line: line number of defect 1 in the patch
suggestion: suggestion for defect 1
</defect>
<defect>
file_path: file path of defect 2 in the patch
line: line number of defect 2 in the patch
suggestion: suggestion for defect 2
</defect>
...
Note:
- If there is no defect, output "None" in this section.
</review>"""


def create_code_review_text(
    dataset_file: Path | str,
    output_dir: Path | str,
    file_source: Literal["oracle", "bm25", "all"],
    k: int | None = None,
    retrieval_file: Path | None = None,
    tokens: list[str] | None = None,
    jobs: int = 2,
) -> None:
    """
    Generate text datasets from SWE-CARE with specified prompts and context sources.

    Args:
        dataset_file: Path to the input SWE-CARE dataset
        output_dir: Directory to save the generated text dataset
        file_source: Source strategy for files - 'oracle', 'bm25', or 'all'
        k: Maximum number of files to use for retrieval
        retrieval_file: File with BM25 retrieval results (required for bm25 file_source)
        tokens: GitHub API tokens (optional)
        jobs: Number of parallel jobs for multithreaded processing (default: 2)
    """
    logger.info(
        f"Starting create_code_review_text with file_source={file_source}, jobs={jobs}"
    )

    if file_source == "bm25":
        # TODO
        raise NotImplementedError("BM25 file source is not implemented yet")

    # Validate arguments
    if file_source == "bm25" and retrieval_file is None:
        raise ValueError("--retrieval_file is required when --file-source is 'bm25'")

    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    if file_source == "bm25" and retrieval_file and not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_file}")

    # Load the dataset
    instances = load_code_review_dataset(dataset_file)

    # Load retrieval results if using bm25
    retrieval_data = None
    if file_source == "bm25" and retrieval_file:
        logger.info(f"Loading BM25 retrieval results from {retrieval_file}")
        try:
            with open(retrieval_file, "r") as f:
                retrieval_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in retrieval file: {e}")

    # Process dataset and generate text
    success_count = 0
    failed_count = 0

    # Create output file and prepare for continuous writing
    output_file = output_dir / f"{dataset_file.stem}__{file_source}.jsonl"
    logger.info(f"Will save processed instances to {output_file}")

    # File lock for thread-safe writing
    file_lock = Lock()

    with open(output_file, "w") as f, ThreadPoolExecutor(max_workers=jobs) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(
                create_code_review_text_instance,
                instance,
                file_source,
                k,
                retrieval_data,
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
    file_source: Literal["oracle", "bm25", "all"],
    k: int | None = None,
    retrieval_data: dict | None = None,
    tokens: list[str] | None = None,
) -> CodeReviewInferenceInstance:
    """
    Process a single instance from the dataset.

    Args:
        instance: Single instance from the dataset
        file_source: Source strategy for files
        k: Maximum number of files to use
        retrieval_data: BM25 retrieval data (if applicable)

    Returns:
        Processed instance with text content
    """
    # Get files based on the source strategy
    if file_source == "oracle":
        files = get_oracle_files(instance, tokens)
    elif file_source == "bm25":
        files = get_bm25_files(instance, retrieval_data, k)
    elif file_source == "all":
        files = get_all_files(instance, k)
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
    instance: CodeReviewTaskInstance, retrieval_data: dict | None, k: int
) -> dict[str, str]:
    """Get files using BM25 retrieval results."""
    # TODO
    raise NotImplementedError("BM25 file source is not implemented yet")


def get_all_files(instance: CodeReviewTaskInstance, k: int) -> dict[str, str]:
    """Get all available files up to k limit."""
    # TODO
    raise NotImplementedError("All file source is not implemented yet")


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
    if not files:
        return ""

    prompt = PROMPT_PREFIX.format(problem_statement=instance.problem_statement)

    prompt += "<code>\n"
    for file_path, file_content in files.items():
        prompt += f"[start of {file_path}]\n"

        if add_line_numbers:
            lines = file_content.split("\n")
            numbered_lines = [f"{i + 1:4d} {line}" for i, line in enumerate(lines)]
            numbered_content = "\n".join(numbered_lines)
            prompt += f"{numbered_content}\n"
        else:
            prompt += f"{file_content}\n"

        prompt += f"[end of {file_path}]\n"
    prompt += "</code>\n"

    prompt += f"<patch>\n{instance.commit_to_review.patch_to_review}\n</patch>\n"

    prompt += REVIEW_PROMPT

    return prompt

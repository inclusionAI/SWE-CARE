"""
Module for converting PR classification data to reward model training samples.
"""

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from tqdm import tqdm

from swe_care.schema.collect import (
    LabeledReviewComment,
    PRClassification,
    RewardModelTrainingSample,
    RewardModelTrainingSampleMetadata,
)
from swe_care.utils.extract_prs_data import (
    extract_problem_statement,
)
from swe_care.utils.file_source_retrieval import (
    get_changed_files_in_patch,
    get_relevant_files,
)
from swe_care.utils.prompt_loader import load_prompt


def convert_to_rm_samples(
    graphql_prs_data_file: Path,
    pr_classification_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
    file_source: Literal[
        "none",
        "base_changed_files",
        "reviewed_file",
        "retrieved_base_changed_files",
        "retrieved_all_files",
    ] = "none",
    jobs: int = 2,
    retrieval_max_files: int = 5,
    retrieval_output_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> None:
    """
    Convert PR classification data to reward model training samples.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file or directory containing *_graphql_prs_data.jsonl files
        pr_classification_file: Path to PR classification file or directory containing *_pr_classification.jsonl files
        output_dir: Output directory for reward model samples
        tokens: Optional list of GitHub tokens for API requests
        file_source: Source for file content ('none', 'base_changed_files', 'reviewed_file', 'retrieved_base_changed_files', or 'retrieved_all_files')
        jobs: Number of parallel jobs
        retrieval_max_files: Maximum number of files to use for retrieval when file_source is 'retrieved_base_changed_files' or 'retrieved_all_files'
        retrieval_output_dir: Output directory for retrieval operations when file_source is 'retrieved_all_files' (required when file_source is 'retrieved_all_files')
        skip_existing: Skip processing existing PR (identified by PR number) in existing repo
    """
    # Validate retrieval_output_dir requirement
    if file_source == "retrieved_all_files" and retrieval_output_dir is None:
        raise ValueError(
            "retrieval_output_dir is required when file_source is 'retrieved_all_files'"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if inputs are files or directories
    if graphql_prs_data_file.is_file() and pr_classification_file.is_file():
        # Single file processing
        logger.info(
            f"Processing single file pair: {graphql_prs_data_file} & {pr_classification_file}"
        )
        samples_count = convert_to_rm_samples_single_file(
            graphql_prs_data_file=graphql_prs_data_file,
            pr_classification_file=pr_classification_file,
            output_dir=output_dir,
            tokens=tokens,
            file_source=file_source,
            retrieval_max_files=retrieval_max_files,
            retrieval_output_dir=retrieval_output_dir,
            skip_existing=skip_existing,
        )
        logger.info(f"Completed: {samples_count} RM samples generated")
    elif graphql_prs_data_file.is_dir() and pr_classification_file.is_dir():
        # Batch processing
        logger.info(
            f"Processing directories: {graphql_prs_data_file} & {pr_classification_file}"
        )

        # Find all GraphQL PRs data files
        graphql_files = list(graphql_prs_data_file.rglob("*_graphql_prs_data.jsonl"))
        if not graphql_files:
            logger.warning(
                f"No GraphQL PRs data files found in {graphql_prs_data_file}"
            )
            return

        # Find matching pairs
        file_pairs = []
        for graphql_file in graphql_files:
            classification_file = find_matching_classification_file(
                graphql_file, pr_classification_file
            )
            if classification_file:
                file_pairs.append((graphql_file, classification_file))
            else:
                logger.warning(
                    f"No matching classification file found for {graphql_file}"
                )

        if not file_pairs:
            logger.warning("No matching file pairs found")
            return

        logger.info(f"Found {len(file_pairs)} file pairs to process")

        # Choose executor based on file_source
        # Use ProcessPoolExecutor for retrieved_all_files to enable better parallelism
        # with file system operations and git worktrees
        if file_source == "retrieved_all_files":
            executor_class = ProcessPoolExecutor
            logger.info(
                f"Using ProcessPoolExecutor with {jobs} workers for retrieved_all_files"
            )
        else:
            executor_class = ThreadPoolExecutor
            logger.info(f"Using ThreadPoolExecutor with {jobs} threads")

        # Process files in parallel
        with executor_class(max_workers=jobs) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    convert_to_rm_samples_single_file,
                    graphql_prs_data_file=graphql_file,
                    pr_classification_file=classification_file,
                    output_dir=output_dir,
                    tokens=tokens,
                    file_source=file_source,
                    retrieval_max_files=retrieval_max_files,
                    retrieval_output_dir=retrieval_output_dir,
                    skip_existing=skip_existing,
                ): (graphql_file, classification_file)
                for graphql_file, classification_file in file_pairs
            }

            # Calculate total PRs to process (estimate based on file sizes)
            total_prs_estimate = 0
            for graphql_file, _ in file_pairs:
                with open(graphql_file, "r") as f:
                    total_prs_estimate += sum(1 for _ in f)

            # Process completed tasks with progress bar
            executor_type = (
                "processes" if file_source == "retrieved_all_files" else "threads"
            )
            with tqdm(
                total=total_prs_estimate,
                desc=f"Converting to RM samples ({jobs} {executor_type})",
                unit="PR",
            ) as pbar:
                successful_files = 0
                failed_files = 0
                total_samples_generated = 0
                prs_processed = 0

                for future in as_completed(future_to_file):
                    graphql_file, classification_file = future_to_file[future]
                    samples_count = 0

                    # Count PRs in this file for progress update
                    with open(graphql_file, "r") as f:
                        file_prs = sum(1 for _ in f)

                    try:
                        samples_count = future.result()
                        successful_files += 1
                        total_samples_generated += samples_count
                        logger.debug(f"Successfully processed {graphql_file}")
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Failed to process {graphql_file}: {e}")

                    prs_processed += file_prs
                    pbar.update(file_prs)
                    pbar.set_postfix(
                        {
                            "files": f"{successful_files}/{len(file_pairs)}",
                            "samples": total_samples_generated,
                        }
                    )

                logger.info(
                    f"Conversion complete: {successful_files} successful, {failed_files} failed"
                )
                logger.info(
                    f"Total generated: {total_samples_generated} RM samples from {prs_processed} PRs"
                )
    else:
        raise ValueError(
            "Both graphql_prs_data_file and pr_classification_file must be either files or directories"
        )


def find_matching_classification_file(
    graphql_data_file: Path, classification_dir: Path
) -> Path | None:
    """
    Find the corresponding classification file for a given graphql data file.

    Args:
        graphql_data_file: Path to a graphql data file (e.g., org__repo_graphql_prs_data.jsonl)
        classification_dir: Directory containing classification files

    Returns:
        Path to the matching classification file, or None if not found
    """
    # Extract repo information from the graphql data file name
    # Expected format: {org}__{repo}_graphql_prs_data.jsonl
    file_name = graphql_data_file.stem
    if file_name.endswith("_graphql_prs_data"):
        repo_part = file_name[: -len("_graphql_prs_data")]
        expected_classification_file = f"{repo_part}_pr_classification.jsonl"

        # Look for the classification file in the classification directory
        classification_file = classification_dir / expected_classification_file
        if classification_file.exists():
            return classification_file

        # Also try recursive search in case files are in subdirectories
        matches = list(classification_dir.rglob(expected_classification_file))
        if matches:
            return matches[0]

    return None


def convert_to_rm_samples_single_file(
    graphql_prs_data_file: Path,
    pr_classification_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
    file_source: Literal[
        "none",
        "base_changed_files",
        "reviewed_file",
        "retrieved_base_changed_files",
        "retrieved_all_files",
    ] = "none",
    retrieval_max_files: int = 5,
    retrieval_output_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> int:
    """
    Convert PR classification data for a single file to reward model training samples.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file
        pr_classification_file: Path to PR classification file
        output_dir: Output directory
        tokens: Optional list of GitHub tokens for API requests
        file_source: Source for file content ('none', 'base_changed_files', 'reviewed_file', 'retrieved_base_changed_files', or 'retrieved_all_files')
        retrieval_max_files: Maximum number of files to use for retrieval when file_source is 'retrieved_base_changed_files' or 'retrieved_all_files'
        retrieval_output_dir: Output directory for retrieval operations when file_source is 'retrieved_all_files' (required when file_source is 'retrieved_all_files')
        skip_existing: Skip processing existing PR (identified by PR number) in existing repo

    Returns:
        Number of RM samples generated
    """
    # Extract repo info from filename
    filename = pr_classification_file.stem
    if not filename.endswith("_pr_classification"):
        raise ValueError(f"Invalid filename format: {filename}")

    repo_part = filename.replace("_pr_classification", "")
    if "__" not in repo_part:
        raise ValueError(
            f"Cannot extract repo owner and name from filename: {filename}"
        )

    repo_owner, repo_name = repo_part.split("__", 1)

    # Create output file
    output_filename = f"{repo_owner}__{repo_name}_rm_samples.jsonl"
    output_file = output_dir / output_filename

    logger.info(f"Processing {repo_owner}/{repo_name} -> {output_file}")

    # Load existing PR numbers from output file if skip_existing is True
    existing_pr_numbers: set[int] = set()
    if skip_existing and output_file.exists():
        logger.info(f"Loading existing PR numbers from {output_file}")
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = RewardModelTrainingSample.from_json(line.strip())
                        existing_pr_numbers.add(sample.metadata.pr_number)
                    except Exception as e:
                        logger.warning(f"Failed to parse line in {output_file}: {e}")
        logger.info(
            f"Found {len(existing_pr_numbers)} existing PR numbers to skip for {repo_owner}/{repo_name}"
        )

    # Load PR classification data
    pr_classifications: list[PRClassification] = []
    with open(pr_classification_file, "r") as f:
        for line in f:
            if line.strip():
                pr_classifications.append(PRClassification.from_json(line.strip()))

    # Create a mapping from PR URL to classification
    pr_classification_map: dict[str, PRClassification] = {
        pr.url: pr for pr in pr_classifications
    }

    # Count total lines for progress bar
    total_lines = 0
    with open(graphql_prs_data_file, "r") as f:
        for _ in f:
            total_lines += 1

    processed_samples = 0

    # Determine file mode based on skip_existing
    file_mode = "a" if skip_existing and output_file.exists() else "w"

    with (
        open(graphql_prs_data_file, "r") as f,
        open(output_file, file_mode) as out_f,
        tqdm(
            total=total_lines,
            desc=f"Processing {repo_owner}/{repo_name}",
            colour="green",
            unit="PR",
        ) as pbar,
    ):
        for line_num, line in enumerate(f, 1):
            try:
                pr_data = json.loads(line.strip())

                # Get PR URL to match with classification data
                pr_url = pr_data.get("url", "")
                if not pr_url or pr_url not in pr_classification_map:
                    pbar.update(1)
                    continue

                pr_classification = pr_classification_map[pr_url]

                # Skip if PR already exists in output file
                if skip_existing and pr_classification.pr_number in existing_pr_numbers:
                    logger.debug(
                        f"Skipping existing PR #{pr_classification.pr_number} for {repo_owner}/{repo_name}"
                    )
                    pbar.update(1)
                    continue

                # Convert PR data and classification to reward model samples
                rm_samples = convert_pr_to_samples(
                    pr_data=pr_data,
                    pr_classification=pr_classification,
                    file_source=file_source,
                    repo=f"{repo_owner}/{repo_name}",
                    tokens=tokens,
                    retrieval_max_files=retrieval_max_files,
                    retrieval_output_dir=retrieval_output_dir,
                )

                # Write samples to output file
                for sample in rm_samples:
                    out_f.write(sample.to_json() + "\n")
                    processed_samples += 1

                # Update progress bar
                pbar.set_postfix({"samples": processed_samples})
                pbar.update(1)

            except Exception as e:
                logger.error(
                    f"Error processing line {line_num} in {graphql_prs_data_file}: {e}"
                )
                pbar.update(1)
                continue

    logger.info(
        f"Processed {processed_samples} reward model samples for {repo_owner}/{repo_name}"
    )

    return processed_samples


def is_positive_review(comment: LabeledReviewComment) -> bool:
    """
    Determine if a review comment is positive.

    A review is positive if referenced_line_changed_in_merged_commit is True and is_resolved is True.

    Args:
        comment: The labeled review comment to check

    Returns:
        True if the review is positive, False otherwise
    """
    return (
        comment.labels.referenced_line_changed_in_merged_commit
        and comment.labels.is_resolved
    )


def convert_pr_to_samples(
    pr_data: dict,
    pr_classification: PRClassification,
    file_source: Literal[
        "none",
        "base_changed_files",
        "reviewed_file",
        "retrieved_base_changed_files",
        "retrieved_all_files",
    ] = "none",
    repo: Optional[str] = None,
    tokens: Optional[list[str]] = None,
    retrieval_max_files: int = 5,
    retrieval_output_dir: Optional[Path] = None,
) -> list[RewardModelTrainingSample]:
    """
    Convert a single PR and its classification to reward model training samples.

    Args:
        pr_data: GraphQL PR data containing closing issues and other PR information
        pr_classification: PR classification data
        file_source: Source for file content ('none', 'base_changed_files', 'reviewed_file', 'retrieved_base_changed_files', or 'retrieved_all_files')
        repo: Repository in format 'owner/name' (needed for file fetching when file_source is 'changed_files')
        tokens: Optional list of GitHub tokens for API requests
        retrieval_max_files: Maximum number of files to use for retrieval when file_source is 'retrieved_base_changed_files' or 'retrieved_all_files'
        retrieval_output_dir: Output directory for retrieval operations when file_source is 'retrieved_all_files' (required when file_source is 'retrieved_all_files')

    Returns:
        List of reward model training samples
    """
    samples: list[RewardModelTrainingSample] = []

    # Extract problem statement from closing issues using the utility function
    closing_issues = pr_data.get("closingIssuesReferences", {}).get("nodes", [])
    problem_statement = extract_problem_statement(closing_issues)

    # If no problem statement from issues, use PR title and body as fallback
    if not problem_statement.strip():
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        problem_statement = f"{title}\n{body}".strip()

    # Extract metadata from PR data and classification
    pr_url = pr_classification.url
    pr_number = pr_classification.pr_number
    # Get base commit
    base_commit = pr_data.get("baseRefOid", "")

    # Extract repository from URL or classification data
    repo = f"{pr_classification.repo_owner}/{pr_classification.repo_name}"

    for commit_classification in pr_classification.commits:
        if not commit_classification.labeled_review_comments:
            # Skip commits without review comments
            continue

        # Use the patch as patch_to_review
        patch_to_review = commit_classification.patch

        if not patch_to_review.strip():
            # Skip if no patch content
            continue

        # First pass: Check if we have both positive and negative reviews
        has_positive = False
        has_negative = False

        for comment in commit_classification.labeled_review_comments:
            if is_positive_review(comment):
                has_positive = True
            else:
                has_negative = True

            # Early exit if we found both types
            if has_positive and has_negative:
                break

        # Skip this commit if we don't have both positive and negative reviews
        if not (has_positive and has_negative):
            continue

        # Now we know we'll create a sample, so we can fetch files
        # Separate positive and negative reviews
        pos_reviews = []
        neg_reviews = []

        # Fetch changed files once if needed for certain file_source strategies to avoid re-fetching
        if file_source in (
            "base_changed_files",
            "retrieved_base_changed_files",
        ):
            changed_files = get_changed_files_in_patch(
                repo=repo,
                base_commit=base_commit,
                patch_to_review=patch_to_review,
                tokens=tokens,
            )
        else:
            changed_files = None

        # Second pass: Process comments with file fetching
        for comment in commit_classification.labeled_review_comments:
            # Get relevant files using the utility function
            relevant_files = get_relevant_files(
                review_comment=comment,
                file_source=file_source,
                repo=repo,
                base_commit=base_commit,
                patch_to_review=patch_to_review,
                tokens=tokens,
                retrieval_max_files=retrieval_max_files,
                retrieval_output_dir=retrieval_output_dir,
                changed_files=changed_files,  # Pass pre-computed changed files to avoid re-fetching
            )

            # Format the review comment using the specified template
            review_str = format_review_comment(
                comment,
                relevant_files,
            )

            if is_positive_review(comment):
                pos_reviews.append(review_str)
            else:
                neg_reviews.append(review_str)

        # Create sample (we already checked that both pos_reviews and neg_reviews exist)
        # Create metadata for this sample
        metadata = RewardModelTrainingSampleMetadata(
            repo=repo,
            pr_number=pr_number,
            url=pr_url,
            commit_to_review=commit_classification.commit_sha,
            file_source=file_source,
        )

        sample = RewardModelTrainingSample(
            problem_statement=problem_statement,
            patch_to_review=patch_to_review,
            pos_review=pos_reviews,
            neg_review=neg_reviews,
            metadata=metadata,
        )
        samples.append(sample)

    return samples


def format_review_comment(
    comment: LabeledReviewComment,
    relevant_files: dict[str, str],
    add_line_numbers: bool = True,
) -> str:
    """
    Format a review comment using the specified template.

    Args:
        comment: LabeledReviewComment object
        relevant_files: Dictionary of relevant files
        add_line_numbers: Whether to add line numbers to the file content

    Returns:
        Formatted review comment string
    """
    # Extract diff_hunk, defaulting to empty string if None
    diff_hunk = comment.diff_hunk or ""

    # Extract path
    path = comment.path or ""

    # Determine line number to use (prioritize original_line, then line, etc.)
    line = comment.original_line
    if line is None:
        line = comment.line
    if line is None:
        line = comment.start_line
    if line is None:
        line = comment.original_start_line
    if line is None:
        line = ""

    # Extract review comment text
    review_comment = comment.text.strip() or ""

    # Load and render the prompt with the provided context
    return load_prompt(
        "rm_sample",
        relevant_files=relevant_files,
        diff_hunk=diff_hunk,
        path=path,
        line=line,
        review_comment=review_comment,
        add_line_numbers=add_line_numbers,
    )

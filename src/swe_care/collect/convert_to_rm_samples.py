"""
Module for converting PR classification data to reward model training samples.
"""

import json
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from swe_care.schema.collect import (
    LabeledReviewComment,
    PRClassification,
    RewardModelTrainingSample,
    RewardModelTrainingSampleMetadata,
)
from swe_care.utils.extract_prs_data import (
    extract_problem_statement,
    fetch_repo_file_content,
    fetch_repo_files_content_by_retrieval,
)
from swe_care.utils.patch import get_changed_file_paths


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
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if inputs are files or directories
    if graphql_prs_data_file.is_file() and pr_classification_file.is_file():
        # Single file processing
        logger.info(
            f"Processing single file pair: {graphql_prs_data_file} & {pr_classification_file}"
        )
        convert_to_rm_samples_single_file(
            graphql_prs_data_file=graphql_prs_data_file,
            pr_classification_file=pr_classification_file,
            output_dir=output_dir,
            tokens=tokens,
            file_source=file_source,
            retrieval_max_files=retrieval_max_files,
        )
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

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=jobs) as executor:
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
                ): (graphql_file, classification_file)
                for graphql_file, classification_file in file_pairs
            }

            # Process completed tasks with progress bar
            with tqdm(
                total=len(file_pairs),
                desc=f"Converting to RM samples ({jobs} threads)",
            ) as pbar:
                successful_files = 0
                failed_files = 0

                for future in as_completed(future_to_file):
                    graphql_file, classification_file = future_to_file[future]

                    try:
                        future.result()
                        successful_files += 1
                        logger.debug(f"Successfully processed {graphql_file}")
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Failed to process {graphql_file}: {e}")

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "success": successful_files,
                            "failed": failed_files,
                        }
                    )

                logger.info(
                    f"Conversion complete: {successful_files} successful, {failed_files} failed"
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
) -> None:
    """
    Convert PR classification data for a single file to reward model training samples.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file
        pr_classification_file: Path to PR classification file
        output_dir: Output directory
        tokens: Optional list of GitHub tokens for API requests
        file_source: Source for file content ('none', 'base_changed_files', 'reviewed_file', 'retrieved_base_changed_files', or 'retrieved_all_files')
        retrieval_max_files: Maximum number of files to use for retrieval when file_source is 'retrieved_base_changed_files' or 'retrieved_all_files'
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

    with (
        open(graphql_prs_data_file, "r") as f,
        open(output_file, "w") as out_f,
        tqdm(
            total=total_lines,
            desc=f"Processing {repo_owner}/{repo_name}",
            colour="blue",
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

                # Convert PR data and classification to reward model samples
                rm_samples = convert_pr_to_samples(
                    pr_data=pr_data,
                    pr_classification=pr_classification,
                    file_source=file_source,
                    repo=f"{repo_owner}/{repo_name}",
                    tokens=tokens,
                    retrieval_max_files=retrieval_max_files,
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

        # Separate positive and negative reviews
        pos_reviews = []
        neg_reviews = []

        if (
            file_source == "base_changed_files"
            or file_source == "retrieved_base_changed_files"
        ):
            changed_files = get_changed_files(
                repo=repo,
                base_commit=base_commit,
                patch_to_review=patch_to_review,
                tokens=tokens,
            )
        else:
            changed_files = {}

        for comment in commit_classification.labeled_review_comments:
            # A review is positive if referenced_line_changed_in_merged_commit is True and is_resolved is True
            is_positive = (
                comment.labels.referenced_line_changed_in_merged_commit
                and comment.labels.is_resolved
            )

            # For "reviewed_file" option, fetch the specific file that this comment applies to
            relevant_files = {}
            if file_source == "reviewed_file" and comment.path:
                try:
                    content = fetch_repo_file_content(
                        repo, base_commit, comment.path, tokens
                    )
                    relevant_files[comment.path] = content
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch content for {comment.path}, ignoring it: {e}"
                    )

            elif file_source == "base_changed_files":
                relevant_files = changed_files

            elif file_source == "retrieved_base_changed_files":
                # If file_source is 'retrieved_files', adopt BM25 to retrieve files that the reviewed file is similar to from the repo
                def preprocess(text):
                    # Convert text to lowercase, remove punctuation marks, and then segment words into tokens.
                    text = text.lower()
                    text = text.translate(str.maketrans("", "", string.punctuation))
                    return word_tokenize(text)

                try:
                    retrieved_files = changed_files
                    # Use BM25 or similar logic to filter files based on the diff_hunk
                    query = comment.diff_hunk
                    # Preprocess the query
                    query_tokens = preprocess(query)
                    # Use BM25 to rank files based on the query
                    bm25 = BM25Okapi(
                        [preprocess(content) for content in retrieved_files.values()]
                    )
                    doc_scores = bm25.get_scores(query_tokens)
                    top_indices = sorted(
                        range(len(doc_scores)),
                        key=lambda i: doc_scores[i],
                        reverse=True,
                    )[: min(retrieval_max_files, len(retrieved_files))]
                    file_paths = list(retrieved_files.keys())
                    # Map the relevant file paths to their contents
                    for idx in top_indices:
                        file_path = file_paths[idx]
                        relevant_files[file_path] = retrieved_files[file_path]
                except Exception as e:
                    logger.warning(f"Failed to retrieved content for diff hunk: {e}")
                    relevant_files = {}

            elif file_source == "retrieved_all_files":
                if comment.diff_hunk:
                    relevant_files = fetch_repo_files_content_by_retrieval(
                        repo=repo,
                        commit=base_commit,
                        query=comment.diff_hunk,
                        tokens=tokens,
                        max_files=retrieval_max_files,
                    )
                else:
                    relevant_files = {}

            else:
                relevant_files = {}

            # Format the review comment using the specified template
            review_str = format_review_comment(
                comment,
                relevant_files,
            )

            if is_positive:
                pos_reviews.append(review_str)
            else:
                neg_reviews.append(review_str)

        # Only create sample if we have both positive and negative reviews
        if pos_reviews and neg_reviews:
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


def get_changed_files(
    repo: str,
    base_commit: str,
    patch_to_review: str,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get file path and file content using changed files strategy.
    Changed files are the changed files in `diff(base_commit, commit_to_review)`.
    """
    changed_files = {}

    logger.debug(f"Getting changed file paths from {base_commit} to commit_to_review")
    changed_file_paths = get_changed_file_paths(patch_to_review)
    logger.debug(f"Changed file paths: {changed_file_paths}")

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

    logger.info(f"Retrieved {len(result)} changed files")
    return result


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

    prompt = ""

    # Format using the specified template
    if relevant_files:
        prompt += "<code>\n"
        for file_path, file_content in relevant_files.items():
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

        prompt += f"<diff_hunk>\n{diff_hunk}\n</diff_hunk>\n<path>{path}</path>\n<line>{line}</line>\n<review_comment>\n{review_comment}\n</review_comment>"
    else:
        # Use current template without file content
        prompt = f"<diff_hunk>\n{diff_hunk}\n</diff_hunk>\n<path>{path}</path>\n<line>{line}</line>\n<review_comment>\n{review_comment}\n</review_comment>"

    return prompt

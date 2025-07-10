"""
Module for classifying review comments as positive or negative samples.

This module analyzes review comments in pull requests and labels them based on
whether the lines they refer to were actually changed in the merged commit.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from swe_care.schema.collect import (
    CommitWithLabeledReviewComments,
    PRCommitWithLabeledReviewComments,
)
from swe_care.utils.extract_prs_data import (
    extract_labeled_review_comments_by_commit,
)


def classify_review_comments(
    graphql_prs_data_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
    jobs: int = 2,
) -> None:
    """
    Classify review comments as positive or negative samples.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file or directory
        output_dir: Output directory for classification results
        tokens: GitHub API tokens
        jobs: Number of parallel jobs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if it's a single file or directory
    if graphql_prs_data_file.is_file():
        # Single file processing
        logger.info(f"Processing single file: {graphql_prs_data_file}")
        classify_review_comments_single_file(
            graphql_prs_data_file=graphql_prs_data_file,
            output_dir=output_dir,
            tokens=tokens,
        )
    elif graphql_prs_data_file.is_dir():
        # Batch processing
        logger.info(f"Processing directory: {graphql_prs_data_file}")

        # Find all GraphQL PRs data files
        graphql_files = list(graphql_prs_data_file.rglob("*_graphql_prs_data.jsonl"))

        if not graphql_files:
            logger.warning(
                f"No GraphQL PRs data files found in {graphql_prs_data_file}"
            )
            return

        logger.info(f"Found {len(graphql_files)} GraphQL PRs data files")

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    classify_review_comments_single_file,
                    graphql_prs_data_file=file_path,
                    output_dir=output_dir,
                    tokens=tokens,
                ): file_path
                for file_path in graphql_files
            }

            # Process completed tasks with progress bar
            with tqdm(
                total=len(graphql_files),
                desc=f"Classifying review comments ({jobs} threads)",
            ) as pbar:
                successful_files = 0
                failed_files = 0

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]

                    try:
                        future.result()
                        successful_files += 1
                        logger.debug(f"Successfully processed {file_path}")
                    except Exception as e:
                        failed_files += 1
                        logger.error(f"Failed to process {file_path}: {e}")

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "success": successful_files,
                            "failed": failed_files,
                        }
                    )

                logger.info(
                    f"Classification complete: {successful_files} successful, {failed_files} failed"
                )
    else:
        raise ValueError(f"Invalid path: {graphql_prs_data_file}")


def classify_review_comments_single_file(
    graphql_prs_data_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
) -> None:
    """
    Classify review comments for a single GraphQL PRs data file.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file
        output_dir: Output directory
        tokens: GitHub API tokens
    """
    # Extract repo info from filename
    filename = graphql_prs_data_file.stem
    if not filename.endswith("_graphql_prs_data"):
        raise ValueError(f"Invalid filename format: {filename}")

    repo_part = filename.replace("_graphql_prs_data", "")
    if "__" not in repo_part:
        raise ValueError(
            f"Cannot extract repo owner and name from filename: {filename}"
        )

    repo_owner, repo_name = repo_part.split("__", 1)
    repo = f"{repo_owner}/{repo_name}"

    # Create output file
    output_filename = (
        f"{repo_owner}__{repo_name}_pr_review_comments_classification.jsonl"
    )
    output_file = output_dir / output_filename

    logger.info(f"Processing {repo} -> {output_file}")

    # Count total lines for progress bar
    total_lines = 0
    with open(graphql_prs_data_file, "r") as f:
        for _ in f:
            total_lines += 1

    processed_prs = 0
    total_commits = 0
    total_comments = 0

    with (
        open(graphql_prs_data_file, "r") as f,
        open(output_file, "w") as out_f,
        tqdm(
            total=total_lines,
            desc=f"Processing {repo}",
            colour="green",
            unit="PR",
        ) as pbar,
    ):
        for line_num, line in enumerate(f, 1):
            try:
                pr_data = json.loads(line.strip())

                # Classify review comments for this PR
                pr_result = classify_review_comments_by_pr(
                    pr_data=pr_data,
                    tokens=tokens,
                )

                if pr_result:
                    # Write result to output file
                    out_f.write(pr_result.to_json() + "\n")
                    processed_prs += 1
                    total_commits += len(pr_result.commit_with_labeled_review_comments)
                    for commit_data in pr_result.commit_with_labeled_review_comments:
                        total_comments += len(commit_data.labeled_review_comments)

                # Update progress bar with current stats
                pbar.set_postfix(
                    {
                        "processed": processed_prs,
                        "commits": total_commits,
                        "comments": total_comments,
                    }
                )
                pbar.update(1)

            except Exception as e:
                logger.error(
                    f"Error processing line {line_num} in {graphql_prs_data_file}: {e}"
                )
                pbar.update(1)
                continue

    logger.info(
        f"Processed {processed_prs} PRs, {total_commits} commits, {total_comments} comments for {repo}"
    )


def classify_review_comments_by_pr(
    pr_data: dict[str, Any],
    tokens: Optional[list[str]] = None,
) -> Optional[PRCommitWithLabeledReviewComments]:
    """
    Classify review comments for a single PR.

    Args:
        pr_data: PR data dictionary
        tokens: GitHub API tokens

    Returns:
        PRCommitWithLabeledReviewComments or None if processing fails
    """
    try:
        # Extract basic PR information
        url = pr_data.get("url", "")
        if not url:
            logger.warning("PR data missing URL")
            return None

        # Parse repo from URL like https://github.com/{repo_owner}/{repo_name}/pull/{pull_number}
        url_parts = url.split("/")
        if len(url_parts) < 5:
            logger.warning(f"Invalid URL format: {url}")
            return None

        repo_owner = url_parts[-4]
        repo_name = url_parts[-3]
        repo = f"{repo_owner}/{repo_name}"
        pr_number = pr_data.get("number")

        if not pr_number:
            logger.warning("PR data missing number")
            return None

        # Get merged commit
        merged_commit = pr_data.get("headRefOid", "")
        if not merged_commit:
            logger.warning(f"PR {pr_number} missing headRefOid")
            return None

        # Get all commits
        commits = pr_data.get("commits", {}).get("nodes", [])
        if not commits:
            logger.warning(f"PR {pr_number} has no commits")
            return None

        commit_with_labeled_review_comments = []

        # Process each commit
        for commit_node in commits:
            commit = commit_node.get("commit", {})
            commit_sha = commit.get("oid")

            if not commit_sha:
                continue

            # Get and label review comments for this commit
            labeled_comments = extract_labeled_review_comments_by_commit(
                pr_data=pr_data,
                commit_to_review=commit_sha,
                merged_commit=merged_commit,
                repo=repo,
                tokens=tokens,
            )

            # Only include commits that have review comments
            if labeled_comments:
                commit_data = CommitWithLabeledReviewComments(
                    commit_sha=commit_sha,
                    labeled_review_comments=labeled_comments,
                )
                commit_with_labeled_review_comments.append(commit_data)

        # Only return result if we have commits with review comments
        if commit_with_labeled_review_comments:
            return PRCommitWithLabeledReviewComments(
                repo_owner=repo_owner,
                repo_name=repo_name,
                pr_number=pr_number,
                url=url,
                commit_with_labeled_review_comments=commit_with_labeled_review_comments,
            )

        return None

    except Exception as e:
        logger.error(f"Error classifying review comments for PR: {e}")
        return None

"""
Build code review task dataset.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from swe_care.collect.evaluate_commits import PRCommitEvaluation
from swe_care.schema.dataset import (
    CodeReviewTaskInstance,
    CodeReviewTaskMetadata,
    CommitToReview,
    ResolvedIssue,
)
from swe_care.utils.estimate import (
    estimate_difficulty,
    estimate_problem_domains,
    estimate_review_effort,
)
from swe_care.utils.extract_prs_data import (
    extract_hints,
    extract_problem_statement,
    extract_reference_review_comments,
    fetch_patch_between_commits,
    fetch_pr_patch,
    fetch_repo_language,
)


def select_best_commit_to_review(
    pr_commits_evaluation: PRCommitEvaluation,
) -> str:
    """Choose the best commit to be reviewed.

    For now, returns the commit with the highest total score.
    """
    # Sort commits by total score in descending order
    sorted_commits = sorted(
        pr_commits_evaluation.commits, key=lambda x: x.total_score, reverse=True
    )

    return sorted_commits[0].commit_sha


def load_existing_instance_ids(output_file: Path) -> set[str]:
    """
    Load existing instance IDs from the output file.

    Args:
        output_file: Path to the output file

    Returns:
        Set of existing instance IDs
    """
    existing_ids = set()
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        instance_id = data.get("instance_id")
                        if instance_id:
                            existing_ids.add(instance_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading existing instances from {output_file}: {e}")

    return existing_ids


def build_code_review_dataset_single_file(
    graphql_prs_data_file: Path,
    pr_commits_evaluation_file: Path,
    output_file: Path,
    existing_instances: set[str],
    file_lock: threading.Lock,
    tokens: Optional[list[str]] = None,
    skip_existing: bool = False,
) -> None:
    """
    Build code review task dataset for a single pair of files.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file (output from get_graphql_prs_data)
        pr_commits_evaluation_file: Path to PR commits evaluation file (output from evaluate_commits)
        output_file: Path to the output file
        existing_instances: Set of existing instance IDs to check against
        file_lock: Threading lock for file operations
        tokens: Optional list of GitHub tokens for API requests
        skip_existing: If True, skip processing existing instance_id.
                      If False, replace existing instance_id data.
    """
    _pr_commits_evaluation: list[PRCommitEvaluation] = [
        PRCommitEvaluation.from_json(e)
        for e in pr_commits_evaluation_file.read_text().split("\n")
        if e
    ]

    pr_commits_evaluation_map: dict[str, PRCommitEvaluation] = {
        e.url: e for e in _pr_commits_evaluation
    }

    # Count the number of lines in the input file
    with open(graphql_prs_data_file, "r") as input_f:
        total_lines = sum(1 for _ in input_f)

    processed_count = 0
    skipped_count = 0

    with open(graphql_prs_data_file, "r") as input_f:
        for line in tqdm(
            input_f, desc=f"Processing {graphql_prs_data_file.name}", total=total_lines
        ):
            try:
                pr_data = json.loads(line.strip())

                # Extract repo name from URL
                url = pr_data.get("url", "")
                if not url:
                    logger.warning("Skipping PR without URL")
                    continue

                # Parse repo from URL like https://github.com/{repo_owner}/{repo_name}/pull/{pull_number}
                url_parts = url.split("/")
                if len(url_parts) < 5:
                    logger.warning(f"Invalid URL format: {url}")
                    continue

                repo_owner = url_parts[-4]
                repo_name = url_parts[-3]
                repo = f"{repo_owner}/{repo_name}"

                # Extract basic fields first
                pull_number = pr_data.get("number")
                title = pr_data.get("title", "")
                body = pr_data.get("body", "")
                created_at = pr_data.get("createdAt", "")
                base_commit = pr_data.get("baseRefOid", "")
                merge_commit = pr_data.get("headRefOid", "")

                if url not in pr_commits_evaluation_map:
                    logger.warning(
                        f"PR URL {url} not found in evaluation data, skipping"
                    )
                    continue

                # Select the best commit for review
                head_commit_to_review = select_best_commit_to_review(
                    pr_commits_evaluation_map[url]
                )

                # Generate instance ID early to check if it exists
                instance_id = CodeReviewTaskInstance.generate_instance_id(
                    repo, pull_number, head_commit_to_review
                )

                # Check if instance already exists and handle according to skip_existing flag
                if instance_id in existing_instances:
                    if skip_existing:
                        logger.debug(f"Skipping existing instance {instance_id}")
                        skipped_count += 1
                        continue
                    else:
                        logger.debug(f"Replacing existing instance {instance_id}")

                logger.debug(f"Selected commit to review: {head_commit_to_review}")

                # Find the commit message for the selected commit
                head_commit_message_to_review = ""
                commits = pr_data.get("commits", {}).get("nodes", [])
                for commit_node in commits:
                    commit = commit_node.get("commit", {})
                    if commit.get("oid") == head_commit_to_review:
                        head_commit_message_to_review = commit.get("message", "")
                        break

                # Extract problem statement from closing issues
                closing_issues = pr_data.get("closingIssuesReferences", {}).get(
                    "nodes", []
                )
                problem_statement = extract_problem_statement(closing_issues)

                # Create resolved issues list
                resolved_issues = []
                for issue in closing_issues:
                    resolved_issue = ResolvedIssue(
                        number=issue.get("number", 0),
                        title=issue.get("title", ""),
                        body=issue.get("body", ""),
                    )
                    resolved_issues.append(resolved_issue)

                # Extract hints (comments from issues before the chosen commit)
                hints_text = extract_hints(pr_data, head_commit_to_review)

                # Extract reference review comments for the chosen commit
                reference_review_comments = extract_reference_review_comments(
                    pr_data, head_commit_to_review
                )

                # Extract patches
                patch_to_review = fetch_patch_between_commits(
                    repo, base_commit, head_commit_to_review, tokens
                )
                merged_patch = fetch_pr_patch(repo, pull_number, tokens)

                # Create metadata
                metadata = CodeReviewTaskMetadata(
                    problem_domains=estimate_problem_domains(pr_data),
                    difficulty=estimate_difficulty(pr_data),
                    estimated_review_effort=estimate_review_effort(pr_data),
                )

                # Get language from the repo
                language = fetch_repo_language(repo, tokens)

                # Create CodeReviewTask instance
                task = CodeReviewTaskInstance(
                    instance_id=instance_id,
                    repo=repo,
                    language=language,
                    pull_number=pull_number,
                    title=title,
                    body=body,
                    created_at=created_at,
                    problem_statement=problem_statement,
                    hints_text=hints_text,
                    resolved_issues=resolved_issues,
                    base_commit=base_commit,
                    commit_to_review=CommitToReview(
                        head_commit=head_commit_to_review,
                        head_commit_message=head_commit_message_to_review,
                        patch_to_review=patch_to_review,
                    ),
                    reference_review_comments=reference_review_comments,
                    merged_commit=merge_commit,
                    merged_patch=merged_patch,
                    metadata=metadata,
                )

                # Write to file with thread lock
                with file_lock:
                    with open(output_file, "a") as output_f:
                        output_f.write(task.to_json() + "\n")

                    # Add to existing instances set to avoid duplicates within this run
                    existing_instances.add(instance_id)

                processed_count += 1
                logger.debug(f"Processed instance: {instance_id}")

            except Exception as e:
                logger.error(f"Error processing PR: {e}")
                continue

    logger.info(
        f"Completed processing {graphql_prs_data_file.name}: {processed_count} processed, {skipped_count} skipped"
    )


def find_matching_evaluation_file(
    graphql_data_file: Path,
    evaluation_dir: Path,
) -> Path | None:
    """
    Find the corresponding evaluation file for a given graphql data file.

    Args:
        graphql_data_file: Path to a graphql data file (e.g., org__repo_graphql_prs_data.jsonl)
        evaluation_dir: Directory containing evaluation files

    Returns:
        Path to the matching evaluation file, or None if not found
    """
    # Extract repo information from the graphql data file name
    # Expected format: {org}__{repo}_graphql_prs_data.jsonl
    file_name = graphql_data_file.stem
    if file_name.endswith("_graphql_prs_data"):
        repo_part = file_name[: -len("_graphql_prs_data")]
        expected_eval_file = f"{repo_part}_pr_commits_evaluation.jsonl"

        # Look for the evaluation file in the evaluation directory
        eval_file = evaluation_dir / expected_eval_file
        if eval_file.exists():
            return eval_file

        # Also try recursive search in case files are in subdirectories
        matches = list(evaluation_dir.rglob(expected_eval_file))
        if matches:
            return matches[0]

    return None


def build_code_review_dataset(
    graphql_prs_data_file: Path | str,
    pr_commits_evaluation_file: Path | str,
    output_dir: Path | str = None,
    tokens: Optional[list[str]] = None,
    skip_existing: bool = False,
    jobs: int = 2,
) -> None:
    """
    Build code review task dataset.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file or directory containing *_graphql_prs_data.jsonl files
        pr_commits_evaluation_file: Path to PR commits evaluation file or directory containing *_pr_commits_evaluation.jsonl files
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
        skip_existing: If True, skip processing existing instance_id in the output file.
                      If False, replace existing instance_id data.
        jobs: Number of concurrent jobs/threads to use
    """
    if isinstance(graphql_prs_data_file, str):
        graphql_prs_data_file = Path(graphql_prs_data_file)
    if isinstance(pr_commits_evaluation_file, str):
        pr_commits_evaluation_file = Path(pr_commits_evaluation_file)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if output_dir is None:
        raise ValueError("output_dir is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "code_review_task_instances.jsonl"

    # Load existing instance IDs
    existing_instances = load_existing_instance_ids(output_file)
    logger.info(f"Found {len(existing_instances)} existing instances in output file")

    # Create a thread lock for file operations
    file_lock = threading.Lock()

    # Determine if inputs are files or directories
    if graphql_prs_data_file.is_file() and pr_commits_evaluation_file.is_file():
        # Single file processing
        logger.info(
            f"Processing single file pair: {graphql_prs_data_file} & {pr_commits_evaluation_file}"
        )
        build_code_review_dataset_single_file(
            graphql_prs_data_file,
            pr_commits_evaluation_file,
            output_file,
            existing_instances,
            file_lock,
            tokens,
            skip_existing,
        )
    elif graphql_prs_data_file.is_dir() and pr_commits_evaluation_file.is_dir():
        # Directory processing with recursive search
        logger.info(
            f"Processing directories: {graphql_prs_data_file} & {pr_commits_evaluation_file}"
        )

        graphql_files = list(graphql_prs_data_file.rglob("*_graphql_prs_data.jsonl"))
        if not graphql_files:
            logger.warning(
                f"No *_graphql_prs_data.jsonl files found in {graphql_prs_data_file}"
            )
            return

        # Find matching pairs
        file_pairs = []
        for graphql_file in graphql_files:
            eval_file = find_matching_evaluation_file(
                graphql_file, pr_commits_evaluation_file
            )
            if eval_file:
                file_pairs.append((graphql_file, eval_file))
            else:
                logger.warning(f"No matching evaluation file found for {graphql_file}")

        if not file_pairs:
            logger.warning("No matching file pairs found")
            return

        logger.info(f"Found {len(file_pairs)} file pairs to process with {jobs} jobs")

        # Process file pairs in parallel
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            # Submit all tasks
            future_to_files = {
                executor.submit(
                    build_code_review_dataset_single_file,
                    graphql_file,
                    eval_file,
                    output_file,
                    existing_instances,
                    file_lock,
                    tokens,
                    skip_existing,
                ): (graphql_file, eval_file)
                for graphql_file, eval_file in file_pairs
            }

            # Process completed tasks
            for future in as_completed(future_to_files):
                graphql_file, eval_file = future_to_files[future]
                try:
                    future.result()
                    logger.success(f"Successfully processed {graphql_file.name}")
                except Exception as e:
                    logger.error(f"Error processing {graphql_file.name}: {e}")
    else:
        raise ValueError(
            "Both graphql_prs_data_file and pr_commits_evaluation_file must be either files or directories"
        )

    logger.info("All dataset building completed")
    logger.info(f"Final dataset saved to {output_file}")

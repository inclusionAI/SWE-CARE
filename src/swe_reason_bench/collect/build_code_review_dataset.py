import json
import random
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from swe_reason_bench.schema.dataset import (
    CodeReviewTaskInstance,
    CodeReviewTaskMetadata,
    CommitToReview,
    ResolvedIssue,
)
from swe_reason_bench.utils.estimate import (
    estimate_difficulty,
    estimate_problem_domains,
    estimate_review_effort,
)
from swe_reason_bench.utils.extract_prs_data import (
    extract_hints,
    extract_patch_between_commits,
    extract_pr_patch,
    extract_problem_statement,
    extract_reference_review_comments,
    get_repo_language,
)


def choose_intermediate_commit_as_review_commit(
    commits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose an intermediate commit to be reviewed. For now, returns a random commit."""
    if not commits:
        return {}

    # TODO: Implement a more sophisticated commit selection logic
    # For simplicity, choose a random commit for now
    chosen_commit = random.choice(commits)

    return chosen_commit.get("commit", {})


def build_code_review_dataset(
    graphql_prs_data_file: Path,
    output_dir: Path = None,
    tokens: Optional[list[str]] = None,
    skip_existing: bool = False,
) -> None:
    """
    Build code review task dataset.

    Args:
        graphql_prs_data_file: Path to GraphQL PRs data file (output from get_graphql_prs_data)
        output_dir: Directory to save the output data
        tokens: Optional list of GitHub tokens for API requests
        skip_existing: If True, skip processing existing instance_id in the output file.
                      If False, replace existing instance_id data.
    """
    if output_dir is None:
        raise ValueError("output_dir is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "code_review_task_instances.jsonl"

    # Load existing instance IDs if the output file exists
    existing_instances = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    instance_id = data.get("instance_id")
                    if instance_id:
                        existing_instances[instance_id] = data
                except json.JSONDecodeError:
                    continue

    # Store all instances (existing + new/updated)
    all_instances = existing_instances.copy()

    # Count the number of lines in the input file
    with open(graphql_prs_data_file, "r") as input_f:
        total_lines = sum(1 for _ in input_f)

    with open(graphql_prs_data_file, "r") as input_f:
        for line in tqdm(input_f, desc="Processing PRs", total=total_lines):
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

                # Choose intermediate commit for review
                commits = pr_data.get("commits", {}).get("nodes", [])
                chosen_commit = choose_intermediate_commit_as_review_commit(commits)
                head_commit_to_review = chosen_commit.get("oid", merge_commit)
                head_commit_message_to_review = chosen_commit.get("messageHeadline", "")

                instance_id = CodeReviewTaskInstance.generate_instance_id(
                    repo, pull_number, head_commit_to_review
                )

                # Check if instance already exists and handle according to skip_existing flag
                if instance_id in existing_instances:
                    if skip_existing:
                        logger.info(f"Skipping existing instance {instance_id}")
                        continue
                    else:
                        logger.info(f"Replacing existing instance {instance_id}")

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
                patch_to_review = extract_patch_between_commits(
                    repo, base_commit, head_commit_to_review, tokens
                )
                merged_patch = extract_pr_patch(repo, pull_number, tokens)

                # Create metadata
                metadata = CodeReviewTaskMetadata(
                    problem_domains=estimate_problem_domains(pr_data),
                    difficulty=estimate_difficulty(pr_data),
                    estimated_review_effort=estimate_review_effort(pr_data),
                )

                # Get language from the repo
                language = get_repo_language(repo, tokens)

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
                        reference_review_comments=reference_review_comments,
                    ),
                    merged_commit=merge_commit,
                    merged_patch=merged_patch,
                    metadata=metadata,
                )

                # Store the task data
                all_instances[instance_id] = json.loads(task.to_json())
                logger.info(f"Processed instance: {instance_id}")

            except Exception as e:
                logger.error(f"Error processing PR: {e}")
                continue

    # Write all instances to the output file (this replaces the entire file)
    with open(output_file, "w") as output_f:
        for instance_data in all_instances.values():
            output_f.write(json.dumps(instance_data) + "\n")

    logger.success(f"Code review dataset saved to {output_file}")

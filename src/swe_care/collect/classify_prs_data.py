"""
Module for classifying PR data including commit evaluation and review comment classification.

This module analyzes PRs by:
1. Evaluating commits using heuristic rules to identify the best commits for review
2. Extracting and labeling review comments based on whether referenced lines were changed
3. Combining both results into a comprehensive classification output
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from swe_care.schema.collect import (
    CommitClassificationResult,
    LabeledReviewComment,
    PRClassification,
)
from swe_care.utils.extract_prs_data import (
    extract_labeled_review_comments_by_commit,
    fetch_patch_between_commits,
)


class CommitEvaluator:
    """Evaluates commits using heuristic rules to identify high-quality commits worthy of review."""

    def __init__(self, pr_data: dict[str, Any]):
        """Initialize the evaluator with rule configurations."""
        self.rules_config = {
            # High-impact rules
            "has_resolved_review_comments": {
                "weight": 3.0,
                "method": self._evaluate_has_resolved_review_comments,
            },
            "has_referenced_line_changed_comments": {
                "weight": 3.0,
                "method": self._evaluate_has_referenced_line_changed_comments,
            },
            "exclude_merge_commit": {
                "weight": 3.0,
                "method": self._evaluate_exclude_merge_commit,
            },
            "exclude_base_merge_commit": {
                "weight": 3.0,
                "method": self._evaluate_exclude_base_merge_commit,
            },
            # Medium-impact rules
            "clear_commit_message": {
                "weight": 2.0,
                "method": self._evaluate_clear_commit_message,
            },
            "conventional_commit": {
                "weight": 2.0,
                "method": self._evaluate_conventional_commit,
            },
            "reasonable_commit_size": {
                "weight": 2.0,
                "method": self._evaluate_reasonable_commit_size,
            },
            "has_associated_review_comments": {
                "weight": 1.5,
                "method": self._evaluate_has_associated_review_comments,
            },
            # Lower-impact rules
            "issue_reference": {
                "weight": 1.0,
                "method": self._evaluate_issue_reference,
            },
            "semantic_commit_message": {
                "weight": 1.0,
                "method": self._evaluate_semantic_commit_message,
            },
            "focused_file_changes": {
                "weight": 1.0,
                "method": self._evaluate_focused_file_changes,
            },
            "descriptive_commit_content": {
                "weight": 1.0,
                "method": self._evaluate_descriptive_commit_content,
            },
        }
        self.pr_data = pr_data

    def evaluate_commit(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> tuple[dict[str, bool | float], float]:
        """
        Evaluate a single commit using all heuristic rules.

        Args:
            commit_data: Dictionary containing commit information
            labeled_review_comments: List of labeled review comments for the commit

        Returns:
            Tuple of (rule_results, total_score)
        """
        rule_results = {}

        # Apply each evaluation rule
        for rule_name, config in self.rules_config.items():
            rule_results[rule_name] = config["method"](
                commit_data,
                labeled_review_comments=labeled_review_comments,
            )

        # Calculate total score
        total_score = self._calculate_total_score(rule_results)

        return rule_results, total_score

    def _calculate_total_score(self, rule_results: dict[str, bool | float]) -> float:
        """Calculate weighted total score from rule results."""
        total_score = 0.0
        total_weight = 0.0

        for rule_name, result in rule_results.items():
            weight = self.rules_config[rule_name]["weight"]
            total_weight += weight

            if isinstance(result, bool):
                score = 1.0 if result else 0.0
            else:
                score = float(result)

            total_score += score * weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _evaluate_has_resolved_review_comments(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit has resolved review comments.

        Higher score for commits that have review comments that were resolved,
        indicating they received meaningful feedback and improvements.
        """
        if not labeled_review_comments:
            return False

        # Check if any of the review comments for this commit were resolved
        for comment in labeled_review_comments:
            if comment.labels.is_resolved:
                return True

        return False

    def _evaluate_has_referenced_line_changed_comments(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit has review comments where the referenced lines were changed in the merged commit.

        Higher score for commits that have review comments pointing to lines that were actually changed,
        indicating the comments were addressing real issues that got fixed.
        """
        if not labeled_review_comments:
            return False

        # Check if any of the review comments reference lines that were changed in the merged commit
        for comment in labeled_review_comments:
            if comment.labels.referenced_line_changed_in_merged_commit:
                return True

        return False

    def _evaluate_has_associated_review_comments(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit has any associated review comments.

        Commits that received review comments are likely more significant.
        """
        commit_oid = commit_data.get("commit", {}).get("oid", "")

        reviews = self.pr_data.get("reviews", {}).get("nodes", [])
        for review in reviews:
            review_comments = review.get("comments", {}).get("nodes", [])
            for comment in review_comments:
                original_commit = comment.get("commit")
                if original_commit is not None:
                    original_commit_oid = original_commit.get("oid")
                    if original_commit_oid == commit_oid:
                        return True

        return False

    def _evaluate_exclude_merge_commit(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if this is NOT a merge commit (merge commits should have low scores).

        As per user requirements: lowest score for commits with "Merge branch" or >2 parents.
        """
        message = commit_data.get("commit", {}).get("message", "")
        parent_count = len(
            commit_data.get("commit", {}).get("parents", {}).get("nodes", [])
        )

        # Check for merge commit indicators
        is_merge = (
            parent_count > 1
            or message.lower().startswith("merge branch")
            or message.lower().startswith("merge pull request")
            or "merge remote-tracking branch" in message.lower()
        )

        return not is_merge

    def _evaluate_exclude_base_merge_commit(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if this is NOT a base/merge commit (lowest score for base and merged commits).

        Base commits and final merge commits are typically not worthy of review.
        """
        commit_oid = commit_data.get("commit", {}).get("oid", "")
        base_commit = self.pr_data.get("baseRefOid", "")
        merge_commit = self.pr_data.get("headRefOid", "")

        # Check if this is the base commit or the final merge commit
        is_base_or_merge = commit_oid == base_commit or commit_oid == merge_commit

        return not is_base_or_merge

    def _evaluate_clear_commit_message(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> float:
        """
        Evaluate the clarity of the commit message.

        Returns a score from 0.0 to 1.0 based on message length, structure, and content.
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message:
            return 0.0

        score = 0.0

        # Length check - too short or too long
        if 10 <= len(message) <= 200:
            score += 0.3
        elif len(message) > 200:
            score += 0.1  # Penalize overly long messages

        # Check for proper capitalization
        if message[0].isupper():
            score += 0.2

        # Check for proper sentence structure (no trailing period for single line)
        lines = message.split("\n")
        if len(lines) == 1 and not message.endswith("."):
            score += 0.2
        elif len(lines) > 1:
            score += 0.2  # Multi-line messages are generally good

        # Check for meaningful content (not just generic words)
        generic_words = ["fix", "update", "change", "modify", "improve"]
        if not any(word in message.lower() for word in generic_words):
            score += 0.3

        return min(1.0, score)

    def _evaluate_conventional_commit(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit follows conventional commit format.

        Conventional commits start with type(scope): description
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message:
            return False

        # Regex for conventional commit format: type(scope): description
        conventional_pattern = r"^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\(.+?\))?: .+"
        return bool(re.match(conventional_pattern, message))

    def _evaluate_semantic_commit_message(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit message is semantic and descriptive.

        Checks for action words and descriptive content.
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message:
            return False

        # Look for action words that indicate meaningful change
        action_words = [
            "add",
            "remove",
            "fix",
            "update",
            "implement",
            "refactor",
            "optimize",
            "improve",
            "enhance",
            "create",
            "delete",
            "modify",
        ]

        return any(word in message.lower() for word in action_words)

    def _evaluate_issue_reference(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> bool:
        """
        Evaluate if the commit message references an issue.

        Looks for issue references like #123, fixes #123, closes #123, etc.
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message:
            return False

        # Pattern for issue references
        issue_patterns = [
            r"#\d+",  # Simple #123
            r"(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+#\d+",
            r"(fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved)\s+\S+/\S+#\d+",
        ]

        for pattern in issue_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True

        return False

    def _evaluate_reasonable_commit_size(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> float:
        """
        Evaluate if the commit has a reasonable size (not too small or too large).

        Returns a score from 0.0 to 1.0 based on the number of changed files and lines.
        """
        # Get commit stats
        additions = commit_data.get("commit", {}).get("additions", 0)
        deletions = commit_data.get("commit", {}).get("deletions", 0)
        changed_files = commit_data.get("commit", {}).get("changedFiles", 0)

        total_changes = additions + deletions

        # Ideal range: 10-200 lines changed, 1-10 files
        if 10 <= total_changes <= 200 and 1 <= changed_files <= 10:
            return 1.0
        elif 5 <= total_changes <= 500 and 1 <= changed_files <= 20:
            return 0.7
        elif 1 <= total_changes <= 1000 and 1 <= changed_files <= 50:
            return 0.4
        else:
            return 0.1

    def _evaluate_focused_file_changes(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> float:
        """
        Evaluate if the commit changes are focused on related files.

        Higher scores for commits that change files in the same directory or with similar purposes.
        """
        changed_files = commit_data.get("commit", {}).get("changedFiles", 0)

        if changed_files <= 1:
            return 1.0  # Single file changes are always focused
        elif changed_files <= 3:
            return 0.8  # Small number of files is usually focused
        elif changed_files <= 10:
            return 0.4  # Scattered
        else:
            return 0.1  # Very scattered

    def _evaluate_descriptive_commit_content(
        self,
        commit_data: dict[str, Any],
        labeled_review_comments: list[LabeledReviewComment],
    ) -> float:
        """
        Evaluate if the commit has descriptive content beyond just the message.

        Considers message length, detail, and context.
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message:
            return 0.0

        score = 0.0

        # Multi-line commits are often more descriptive
        lines = message.split("\n")
        if len(lines) > 1:
            score += 0.4

        # Check for detailed explanation
        if len(message) > 100:
            score += 0.3

        # Check for structured content (lists, sections)
        if any(marker in message for marker in ["*", "-", "1.", "2.", "###", "##"]):
            score += 0.3

        return min(1.0, score)


def classify_prs_data(
    graphql_prs_data_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
    jobs: int = 2,
) -> None:
    """
    Classify PR data including commit evaluation and review comment classification.

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
        classify_prs_data_single_file(
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
                    classify_prs_data_single_file,
                    graphql_prs_data_file=file_path,
                    output_dir=output_dir,
                    tokens=tokens,
                ): file_path
                for file_path in graphql_files
            }

            # Process completed tasks with progress bar
            with tqdm(
                total=len(graphql_files),
                desc=f"Classifying PRs data ({jobs} threads)",
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


def classify_prs_data_single_file(
    graphql_prs_data_file: Path,
    output_dir: Path,
    tokens: Optional[list[str]] = None,
) -> None:
    """
    Classify PR data for a single GraphQL PRs data file.

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

    # Create output file with new naming convention
    output_filename = f"{repo_owner}__{repo_name}_pr_classification.jsonl"
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

                # Classify PR data (evaluation + review comments)
                pr_result = classify_pr_data(
                    pr_data=pr_data,
                    tokens=tokens,
                )

                if pr_result:
                    # Write result to output file
                    out_f.write(pr_result.to_json() + "\n")
                    processed_prs += 1
                    total_commits += len(pr_result.commits)
                    for commit_data in pr_result.commits:
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


def classify_pr_data(
    pr_data: dict[str, Any],
    tokens: Optional[list[str]] = None,
) -> Optional[PRClassification]:
    """
    Classify a single PR including commit evaluation and review comment classification.

    Args:
        pr_data: PR data dictionary
        tokens: GitHub API tokens

    Returns:
        PRClassification or None if processing fails
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

        # Get base commit
        base_commit = pr_data.get("baseRefOid", "")
        if not base_commit:
            logger.warning(f"PR {pr_number} missing baseRefOid")
            return None

        # Get all commits
        commits = pr_data.get("commits", {}).get("nodes", [])
        if not commits:
            logger.warning(f"PR {pr_number} has no commits")
            return None

        commit_classification_results = []

        # Process each commit
        evaluator = CommitEvaluator(pr_data=pr_data)
        for commit_node in commits:
            commit = commit_node.get("commit", {})
            commit_sha = commit.get("oid")

            if not commit_sha:
                continue

            # Get and label review comments for this commit
            labeled_review_comments = extract_labeled_review_comments_by_commit(
                pr_data=pr_data,
                commit_to_review=commit_sha,
                merged_commit=merged_commit,
                repo=repo,
                tokens=tokens,
            )

            # Evaluate the commit using heuristic rules
            rule_results, total_score = evaluator.evaluate_commit(
                commit_data=commit_node,
                labeled_review_comments=labeled_review_comments,
            )

            # Fetch patch between base commit and current commit
            patch = fetch_patch_between_commits(
                repo=repo,
                base_commit=base_commit,
                head_commit=commit_sha,
                tokens=tokens,
            )

            # Create combined classification result
            commit_classification = CommitClassificationResult(
                commit_sha=commit_sha,
                labeled_review_comments=labeled_review_comments,
                total_score=total_score,
                rule_results=rule_results,
                patch=patch,
            )
            commit_classification_results.append(commit_classification)

        # Sort commits by total score in descending order (best commits first)
        commit_classification_results.sort(key=lambda x: x.total_score, reverse=True)

        # Return result with all commits and their classification data
        return PRClassification(
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=pr_number,
            url=url,
            commits=commit_classification_results,
        )

    except Exception as e:
        logger.error(f"Error classifying PR data: {e}")
        return None


classify_prs_data(graphql_prs_data_file=Path('results/graphql_prs_data/Significant-Gravitas__AutoGPT_graphql_prs_data.jsonl'),output_dir=Path('./results/classify_prs_data'),tokens=['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'],jobs=1)
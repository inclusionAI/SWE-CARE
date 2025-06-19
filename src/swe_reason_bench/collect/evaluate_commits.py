import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataclasses_json import dataclass_json
from loguru import logger

from swe_reason_bench.utils.extract_prs_data import extract_reference_review_comments


@dataclass_json
@dataclass
class CommitEvaluationResult:
    """Results from evaluating a commit."""

    commit_sha: str
    total_score: float
    rule_results: dict[str, bool | float]
    rule_category: dict[str, str]


@dataclass_json
@dataclass
class PRCommitEvaluation:
    """Results from evaluating a PR."""

    repo_owner: str
    repo_name: str
    pr_number: int
    url: str
    commits: list[CommitEvaluationResult]


class CommitEvaluator:
    """Evaluates commits using heuristic rules to identify high-quality commits worthy of review."""

    def __init__(self):
        """Initialize the evaluator with rule configurations."""
        self.rules_config = {
            # High-impact rules
            "has_resolved_review_comments": {
                "weight": 3.0,
                "category": "review_activity",
                "method": self._evaluate_has_resolved_review_comments,
            },
            "exclude_merge_commit": {
                "weight": 3.0,
                "category": "commit_type",
                "method": self._evaluate_exclude_merge_commit,
            },
            "exclude_base_merge_commit": {
                "weight": 3.0,
                "category": "commit_type",
                "method": self._evaluate_exclude_base_merge_commit,
            },
            # Medium-impact rules
            "clear_commit_message": {
                "weight": 2.0,
                "category": "message_quality",
                "method": self._evaluate_clear_commit_message,
            },
            "conventional_commit": {
                "weight": 2.0,
                "category": "message_format",
                "method": self._evaluate_conventional_commit,
            },
            "reasonable_commit_size": {
                "weight": 2.0,
                "category": "code_quality",
                "method": self._evaluate_reasonable_commit_size,
            },
            "has_associated_review_comments": {
                "weight": 1.5,
                "category": "review_activity",
                "method": self._evaluate_has_associated_review_comments,
            },
            # Lower-impact rules
            "issue_reference": {
                "weight": 1.0,
                "category": "traceability",
                "method": self._evaluate_issue_reference,
            },
            "semantic_commit_message": {
                "weight": 1.0,
                "category": "message_quality",
                "method": self._evaluate_semantic_commit_message,
            },
            "focused_file_changes": {
                "weight": 1.0,
                "category": "code_quality",
                "method": self._evaluate_focused_file_changes,
            },
            "descriptive_commit_content": {
                "weight": 1.0,
                "category": "message_quality",
                "method": self._evaluate_descriptive_commit_content,
            },
        }

    def evaluate_commit(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> CommitEvaluationResult:
        """
        Evaluate a single commit using all heuristic rules.

        Args:
            commit_data: Dictionary containing commit information
            pr_data: Dictionary containing PR information for context

        Returns:
            CommitEvaluationResult with rule results and total score
        """
        commit_sha = commit_data.get("commit", {}).get("oid", "")
        rule_results = {}
        rule_category = {}

        # Apply each evaluation rule
        for rule_name, config in self.rules_config.items():
            rule_results[rule_name] = config["method"](commit_data, pr_data)
            rule_category[rule_name] = config["category"]

        # Calculate total score
        total_score = self._calculate_total_score(rule_results)

        return CommitEvaluationResult(
            commit_sha=commit_sha,
            rule_results=rule_results,
            total_score=total_score,
            rule_category=rule_category,
        )

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
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if the commit has resolved review comments.

        Higher score for commits that have review comments that were resolved,
        indicating they received meaningful feedback and improvements.
        """
        reference_review_comments = extract_reference_review_comments(
            pr_data, commit_data.get("commit", {}).get("oid", "")
        )

        return len(reference_review_comments) > 0

    def _evaluate_has_associated_review_comments(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if the commit has any associated review comments.

        Commits that received review comments are likely more significant.
        """
        commit_oid = commit_data.get("commit", {}).get("oid", "")

        reviews = pr_data.get("reviews", {}).get("nodes", [])
        for review in reviews:
            review_comments = review.get("comments", {}).get("nodes", [])
            for comment in review_comments:
                original_commit_oid = comment.get("originalCommit", {}).get("oid")
                if original_commit_oid == commit_oid:
                    return True

        return False

    def _evaluate_exclude_merge_commit(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
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
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if this is NOT a base/merge commit (lowest score for base and merged commits).

        Base commits and final merge commits are typically not worthy of review.
        """
        commit_oid = commit_data.get("commit", {}).get("oid", "")
        base_commit = pr_data.get("baseRefOid", "")
        head_commit = pr_data.get("headRefOid", "")

        # Base commit and head commit (final merge) should have low scores
        if commit_oid == base_commit or commit_oid == head_commit:
            return False

        return True

    def _evaluate_clear_commit_message(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> float:
        """
        Evaluate if the commit message is clear and descriptive.

        Returns a score between 0.0 and 1.0 based on message quality.
        """
        message = commit_data.get("commit", {}).get("message", "")

        if not message or len(message.strip()) < 10:
            return 0.0

        # Score based on multiple factors
        score = 0.0

        # Length factor (not too short, not too long)
        msg_length = len(message.strip())
        if 20 <= msg_length <= 200:
            score += 0.3
        elif 10 <= msg_length <= 300:
            score += 0.2

        # Meaningful content indicators
        meaningful_words = [
            "fix",
            "add",
            "remove",
            "update",
            "refactor",
            "improve",
            "implement",
            "feature",
        ]
        if any(word in message.lower() for word in meaningful_words):
            score += 0.3

        # Proper capitalization
        if message[0].isupper():
            score += 0.1

        # Contains explanation (has punctuation suggesting detail)
        if any(punct in message for punct in [".", ":", ";"]):
            score += 0.2

        # Avoid generic messages
        generic_patterns = ["fix", "update", "change", "wip", "tmp", "test"]
        if message.lower().strip() in generic_patterns:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _evaluate_conventional_commit(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if the commit follows conventional commit format.

        Conventional commits are well-structured and indicate quality.
        """
        message = commit_data.get("commit", {}).get("message", "")

        # Conventional commit pattern: type(scope): description
        conventional_pattern = r"^(feat|fix|docs|style|refactor|test|chore|build|ci|perf|revert)(\(.+\))?: .+"

        return bool(re.match(conventional_pattern, message, re.IGNORECASE))

    def _evaluate_semantic_commit_message(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if commit message follows semantic patterns (broader than conventional).
        """
        message = commit_data.get("commit", {}).get("message", "")

        # Semantic patterns (including conventional and other good patterns)
        semantic_patterns = [
            r"^(feat|feature|fix|bug|docs?|style|refactor|test|chore|build|ci|perf|revert)[\(:]",
            r"^(add|remove|update|improve|enhance|optimize|implement)[\s:]",
            r"^[A-Z][a-z]+\s+[a-z]",  # Capitalized action word
        ]

        return any(
            re.match(pattern, message, re.IGNORECASE) for pattern in semantic_patterns
        )

    def _evaluate_issue_reference(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> bool:
        """
        Evaluate if the commit references an issue or PR.

        Issue references indicate traceability and planned work.
        """
        message = commit_data.get("commit", {}).get("message", "")

        # Pattern matching for various reference types
        reference_patterns = [
            r"#\d+",  # GitHub issue/PR reference
            r"fixes?\s+#\d+",  # Fix references
            r"closes?\s+#\d+",  # Close references
            r"resolves?\s+#\d+",  # Resolve references
            r"addresses?\s+#\d+",  # Address references
        ]

        return any(
            re.search(pattern, message, re.IGNORECASE) for pattern in reference_patterns
        )

    def _evaluate_reasonable_commit_size(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> float:
        """
        Evaluate if the commit size is reasonable.

        Returns a score between 0.0 and 1.0 based on change size.
        """
        additions = commit_data.get("commit", {}).get("additions", 0)
        deletions = commit_data.get("commit", {}).get("deletions", 0)
        total_changes = additions + deletions

        # Handle missing data
        if total_changes == 0:
            return 0.5  # Neutral score if no size info

        # Optimal range scoring
        if 5 <= total_changes <= 200:
            return 1.0  # Sweet spot for reviewable changes
        elif 200 < total_changes <= 500:
            return 0.8  # Larger but manageable
        elif 1 <= total_changes < 5:
            return 0.6  # Very small changes
        elif 500 < total_changes <= 1000:
            return 0.4  # Large changes
        else:
            return 0.1  # Very large changes

    def _evaluate_focused_file_changes(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
    ) -> float:
        """
        Evaluate if the file changes are focused and coherent.

        Returns a score between 0.0 and 1.0 based on file change patterns.
        """
        changed_files = commit_data.get("commit", {}).get("changedFilesIfAvailable", 0)

        if changed_files == 0:
            return 0.5  # Neutral if no info
        elif changed_files <= 5:
            return 1.0  # Very focused
        elif changed_files <= 15:
            return 0.7  # Moderately focused
        elif changed_files <= 30:
            return 0.4  # Scattered
        else:
            return 0.1  # Very scattered

    def _evaluate_descriptive_commit_content(
        self, commit_data: dict[str, Any], pr_data: dict[str, Any]
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


def evaluate_commits(
    graphql_prs_data_file: Path,
    output_dir: Path,
    tokens: list[str] | None = None,
) -> None:
    """
    Evaluate commits in PRs using heuristic rules.

    Args:
        graphql_prs_data_file: Path to the GraphQL PRs data file
        output_dir: Directory to save evaluation results
        tokens: GitHub tokens (not used in this function but kept for consistency)
    """
    logger.info(f"Starting commit evaluation from {graphql_prs_data_file}")

    evaluator = CommitEvaluator()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track which repositories we've already processed to avoid deleting output files multiple times
    processed_repos = set()

    with open(graphql_prs_data_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                pr_data = json.loads(line.strip())

                # Extract repo name from URL
                url = pr_data.get("url", "")
                if not url:
                    logger.warning(f"Line {line_num}: Skipping PR without URL")
                    continue

                # Parse repo from URL like https://github.com/{repo_owner}/{repo_name}/pull/{pull_number}
                url_parts = url.split("/")
                if len(url_parts) < 5:
                    logger.warning(f"Line {line_num}: Invalid URL format: {url}")
                    continue

                repo_owner = url_parts[-4]
                repo_name = url_parts[-3]
                pr_number = pr_data.get("number")

                # Output file for this repository
                output_file = (
                    output_dir
                    / f"{repo_owner}__{repo_name}_pr_commits_evaluation.jsonl"
                )

                # Delete the output file if this is the first time processing this repository
                repo_key = f"{repo_owner}__{repo_name}"
                if repo_key not in processed_repos:
                    if output_file.exists():
                        output_file.unlink()
                        logger.info(f"Deleted existing output file: {output_file}")
                    processed_repos.add(repo_key)

                # Process commits in this PR
                commits = pr_data.get("commits", {}).get("nodes", [])

                logger.info(
                    f"Processing PR #{pr_number} from {repo_owner}/{repo_name} with {len(commits)} commits"
                )

                pr_evaluation_record = PRCommitEvaluation(
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    pr_number=pr_number,
                    url=url,
                    commits=[],
                )

                for commit in commits:
                    try:
                        evaluation_result = evaluator.evaluate_commit(commit, pr_data)
                        pr_evaluation_record.commits.append(evaluation_result)

                        logger.debug(
                            f"  Evaluated commit {evaluation_result.commit_sha}: "
                            f"score={evaluation_result.total_score:.3f}"
                        )

                    except Exception as e:
                        commit_oid = commit.get("commit", {}).get("oid", "unknown")
                        logger.error(f" Error evaluating commit {commit_oid}: {e}")
                        continue

                # Sort commits by total score in descending order
                pr_evaluation_record.commits.sort(
                    key=lambda x: x.total_score, reverse=True
                )

                # Write evaluation results
                with open(output_file, "a") as out_f:
                    out_f.write(pr_evaluation_record.to_json() + "\n")

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: Invalid JSON: {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Error processing PR: {e}")
                continue

    logger.info("Commit evaluation completed")

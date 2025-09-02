"""
Repo-level LLM evaluator for code review predictions.
"""

import json
import re
from typing import Any, Literal, Optional

from loguru import logger

from swe_care.harness.evaluators import Evaluator
from swe_care.harness.evaluators.utils import extract_defects_from_review
from swe_care.schema.dataset import (
    CodeReviewTaskInstance,
    ReferenceReviewComment,
)
from swe_care.schema.evaluation import CodeReviewPrediction
from swe_care.utils.file_source_retrieval import get_relevant_files
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.prompt_loader import load_prompt

# System prompt is now loaded from the YAML template


class RepoLevelLLMEvaluator(Evaluator):
    """Evaluator that uses LLM to classify review comments as positive or negative."""

    def __init__(
        self,
        model_client: BaseModelClient,
        file_source: Literal[
            "none",
            "base_changed_files",
            "reviewed_file",
            "retrieved_base_changed_files",
            "retrieved_all_files",
        ] = "none",
        retrieval_max_files: int = 5,
        retrieval_output_dir: Optional[str] = None,
        tokens: Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize the Repo-level LLM evaluator.

        Args:
            model_client: The LLM client to use for evaluation
            file_source: Source for file content
            retrieval_max_files: Maximum number of files for retrieval
            retrieval_output_dir: Output directory for retrieval operations
            tokens: GitHub API tokens
        """
        super().__init__(**kwargs)
        self.model_client = model_client
        self.file_source = file_source
        self.retrieval_max_files = retrieval_max_files
        self.retrieval_output_dir = retrieval_output_dir
        self.tokens = tokens

        # Validate retrieval_output_dir requirement
        if file_source == "retrieved_all_files" and retrieval_output_dir is None:
            raise ValueError(
                "retrieval_output_dir is required when file_source is 'retrieved_all_files'"
            )

    @property
    def requires_input(self) -> bool:
        """Need the input to get problem statement and base commit"""
        return True

    @property
    def requires_reference(self) -> bool:
        """Reference is optional for this evaluator"""
        return False

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        # First, try to find JSON string within triple backticks
        # Handle both ```json and ``` formats
        json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(json_pattern, text, re.DOTALL | re.MULTILINE)

        for match in matches:
            try:
                cleaned_match = match.strip()
                if cleaned_match:
                    return json.loads(cleaned_match)
            except json.JSONDecodeError:
                continue

        # If no backtick-wrapped JSON found, try to extract JSON object directly
        # Look for content between { and }
        json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        json_matches = re.findall(json_object_pattern, text, re.DOTALL)

        for match in json_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Last resort: try parsing the entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # If all else fails, provide more detailed error
            logger.error(f"Failed to parse JSON from response: {text[:500]}...")
            raise ValueError("No valid JSON found in LLM response")

    def evaluate_on_reference_review_comment(
        self,
        review_comment: ReferenceReviewComment,
        input: CodeReviewTaskInstance,
    ) -> dict:
        """Evaluate a single reference review comment.

        Args:
            review_comment: The review comment to evaluate
            input: The input task instance

        Returns:
            Dictionary with label (0/1) and reason
        """
        # Get file content based on file_source
        relevant_files = self._get_relevant_files(review_comment, input)

        # Format the review comment with context
        formatted_review = self._format_review_comment_with_context(
            review_comment,
            relevant_files,
        )

        # Load prompts from YAML template
        system_prompt, user_prompt = load_prompt(
            "repo_level_evaluation",
            problem_statement=input.problem_statement,
            patch_to_review=input.commit_to_review.patch_to_review,
            formatted_review=formatted_review,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.model_client.create_completion(messages)
        result = self._parse_json(answer)

        # Validate result
        if "label" not in result or result["label"] not in [0, 1]:
            raise ValueError(f"Invalid label in result: {result}")
        if "reason" not in result:
            result["reason"] = "No reason provided"

        return result

    def _get_relevant_files(
        self,
        review_comment: ReferenceReviewComment,
        input: CodeReviewTaskInstance,
    ) -> dict[str, str]:
        """Get relevant files based on file_source strategy."""
        return get_relevant_files(
            review_comment=review_comment,
            file_source=self.file_source,
            repo=input.repo,
            base_commit=input.base_commit,
            patch_to_review=input.commit_to_review.patch_to_review,
            tokens=self.tokens,
            retrieval_max_files=self.retrieval_max_files,
            retrieval_output_dir=self.retrieval_output_dir,
        )

    def _format_review_comment_with_context(
        self,
        review_comment: ReferenceReviewComment,
        relevant_files: dict[str, str],
    ) -> str:
        """Format review comment with file context."""
        # Load and render the template
        return load_prompt(
            "rm_sample",
            relevant_files=relevant_files,
            diff_hunk=review_comment.diff_hunk or "",
            path=review_comment.path or "",
            line=review_comment.line or "",
            review_comment=review_comment.text.strip(),
            add_line_numbers=True,
        )

    def _evaluate(
        self,
        *,
        prediction: CodeReviewPrediction,
        reference: Any,  # noqa: ARG002
        input: CodeReviewTaskInstance,
    ) -> dict:
        """Evaluate code review prediction by classifying extracted defects.

        Args:
            prediction: The code review prediction
            reference: Not used in this evaluator
            input: The input task instance

        Returns:
            Dictionary containing evaluation metrics
        """
        # Extract defects from the predicted review
        predicted_defects = extract_defects_from_review(prediction.review_text, input)

        if not predicted_defects:
            # No defects found means it's a positive review
            return {
                "score": 1.0,
                "num_defects": 0,
                "classifications": [],
            }

        # Classify each defect
        classifications = []
        positive_count = 0
        negative_count = 0

        for defect in predicted_defects:
            try:
                result = self.evaluate_on_reference_review_comment(defect, input)
                classifications.append(
                    {
                        "defect_text": defect.text[:200] + "..."
                        if len(defect.text) > 200
                        else defect.text,
                        "label": result["label"],
                        "reason": result["reason"],
                    }
                )

                if result["label"] == 1:
                    positive_count += 1
                else:
                    negative_count += 1

            except Exception as e:
                logger.error(f"Failed to classify defect: {e}")
                classifications.append(
                    {
                        "defect_text": defect.text[:200] + "..."
                        if len(defect.text) > 200
                        else defect.text,
                        "label": 0,  # Default to negative on error
                        "reason": f"Classification error: {str(e)}",
                    }
                )
                negative_count += 1

        # Calculate overall score
        total_defects = len(predicted_defects)
        if total_defects > 0:
            score = positive_count / total_defects
        else:
            score = 1.0  # No defects means positive

        return {
            "score": score,
            "num_defects": total_defects,
            "num_positive": positive_count,
            "num_negative": negative_count,
            "classifications": classifications,
        }

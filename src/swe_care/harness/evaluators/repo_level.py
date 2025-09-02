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
from swe_care.utils.template import render_template

REPO_LEVEL_EVALUATION_SYSTEM_PROMPT = """\
You are an expert code review evaluator. Your task is to classify code review comments as positive (useful and high-quality) or negative (not useful or low-quality) based on their contribution to improving the code and solving the problem.

IMPORTANT: This classification is NOT about whether the reviewer approves or disapproves of the code. Instead, it's about whether the review comment itself is valuable and helpful.

A POSITIVE (label: 1) review comment is one that is USEFUL and HIGH-QUALITY. It should have most of these characteristics:

1. **Identifies Real Issues**:
   - Points out actual bugs, logic errors, or edge cases
   - Identifies security vulnerabilities or performance problems
   - Highlights code that doesn't solve the stated problem
   - Finds missing error handling or validation

2. **Provides Actionable Feedback**:
   - Suggests specific improvements with clear reasoning
   - Proposes better algorithms, data structures, or design patterns
   - Recommends more maintainable or readable approaches
   - Points to specific lines or sections needing attention

3. **Ensures Problem Resolution**:
   - Verifies the patch actually addresses the problem statement
   - Identifies gaps between the requirements and implementation
   - Points out incomplete solutions or missing features
   - Suggests additional test cases or scenarios to consider

4. **Improves Code Quality**:
   - Highlights violations of best practices with explanations
   - Suggests refactoring for better modularity or reusability
   - Identifies code smells or anti-patterns
   - Recommends better naming, documentation, or structure

5. **Provides Educational Value**:
   - Explains why certain approaches are problematic
   - Shares domain knowledge or technical insights
   - References relevant documentation or standards
   - Helps the developer learn and improve

A NEGATIVE (label: 0) review comment is one that is NOT USEFUL or LOW-QUALITY:

1. **Generic or Vague**:
   - Simple approval without analysis ("LGTM", "Looks good", "+1")
   - Vague statements without specifics ("This could be better")
   - Comments that don't explain the reasoning

2. **Irrelevant or Off-topic**:
   - Discusses code not changed in the patch
   - Brings up unrelated issues or features
   - Personal preferences without technical merit

3. **Trivial or Nitpicking**:
   - Minor style issues that don't affect functionality
   - Bikeshedding about naming when current names are clear
   - Formatting complaints when code follows project standards

4. **Incorrect or Misleading**:
   - Misunderstands what the code does
   - Suggests changes that would break functionality
   - Based on wrong assumptions about requirements

5. **Unconstructive**:
   - Criticism without suggestions for improvement
   - Dismissive or discouraging without being helpful
   - Focuses on the person rather than the code

Remember: A comment pointing out problems is POSITIVE if it helps improve the code. A comment praising the code is NEGATIVE if it doesn't add value or ensure quality.

You will analyze the review comment in the context of:
1. The problem statement (what needs to be solved)
2. The patch/diff (what changes were made)
3. The file context (surrounding code when available)
4. The specific review comment text

Output your classification as JSON:
```json
{
    "label": <1 for positive/useful, 0 for negative/not-useful>,
    "reason": "<Explain why this comment is or isn't useful and high-quality>"
}
```
"""


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

        # Create user prompt
        user_prompt = f"""Please analyze the following code review comment and classify it as positive (useful/high-quality) or negative (not useful/low-quality).

<issue>
{input.problem_statement}
</issue>

<patch_to_review>
{input.commit_to_review.patch_to_review}
</patch_to_review>

<review>
{formatted_review}
</review>"""

        messages = [
            {"role": "system", "content": REPO_LEVEL_EVALUATION_SYSTEM_PROMPT},
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
        # Use the existing template rendering
        return render_template(
            "rm_sample.j2",
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

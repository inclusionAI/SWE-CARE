import difflib
import json
import re
from typing import Any

from swe_care.harness.evaluators import Evaluator
from swe_care.schema.dataset import (
    CodeReviewTaskInstance,
    ReferenceReviewComment,
)
from swe_care.schema.evaluation import CodeReviewPrediction
from swe_care.utils.llm_models.clients import BaseModelClient

EVALUATION_PROMPT = """\
Your task is to evaluate the quliaty of the code review. Below are the required fields of a standard code review:

- Function: A brief description of the main purpose and implemented functionality of the patch.
- Complexity: An evaluation of whether the patch is more complex than it should be. The evaluation should cover different granularities: line-level, function-level, class-level, and file-level, if applicable.
- Style: An evaluation of whether the patch follows the programming conventions of the original code, e.g. the naming of variables and functions.
- Documentation: An evaluation of whether the patch provide clear and necessary comments, as well as documentation.

For each field, you should analyze:

- Correctness: whether the review is technically correct and contains no factual errors with regard to the provided issue, code base, and patch.
- Relevance: whether the review is targeted at the issue and the code patch.
- Clarity: whether the review is clear and without redundant information.
- Consistency: whether the review is logically consistent with the issue, code base, patch, and other fields in the review.
- Language: whether the review uses professional language and contains no grammatical errors.

Give a score between 0 and 1 (inclusive) to each of these five dimensions, and output your final evaluation in nested json format:
```
{
    "function": {"correctness": score, "relevance": score, "clarity": score, "consistency": score, "language": score},
    "complexity": {"correctness": score, "relevance": score, "clarity": score, "consistency": score, "language": score},
    "style": {"correctness": score, "relevance": score, "clarity": score, "consistency": score, "language": score},
    "documentation": {"correctness": score, "relevance": score, "clarity": score, "consistency": score, "language": score}
}
```

If you cannot identify a certain field from the review, give a 0 score to all dimensions in this field.
"""


class LLMEvaluator(Evaluator):
    """Evaluator for code review predictions against benchmark dataset."""

    def __init__(self, model_client: BaseModelClient, **kwargs):
        """Initialize the LLM evaluator with a model client."""
        super().__init__(**kwargs)
        self.model_client = model_client

    def _parse_json(self, text: str) -> dict:
        # Try to find JSON string within triple backticks, assuming there are possibly multiple json markdown string
        matches = re.finditer(r"```(json)?(.*?)```", text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.group(2))
            except json.JSONDecodeError:
                continue

        raise ValueError(f"No valid JSON found in LLM response: {text}")

    def _evaluate(
        self,
        *,
        prediction: CodeReviewPrediction,
        reference: Any,
        input: CodeReviewTaskInstance,
    ) -> dict:
        messages = [
            {
                "role": "user",
                "content": prediction.review_text + "\n" + EVALUATION_PROMPT,
            }
        ]

        answer = self.model_client.create_completion(messages)
        return self._parse_json(answer)


class RuleBasedEvaluator(Evaluator):
    """Evaluator for code review predictions against benchmark dataset."""

    def __init__(
        self, location_weight: float = 0.5, description_weight: float = 0.5, **kwargs
    ):
        """Initialize the rule-based evaluator.

        Args:
            location_weight: Weight for location similarity in final score (default: 0.5)
            description_weight: Weight for description similarity in final score (default: 0.5)
        """
        super().__init__(**kwargs)
        self.location_weight = location_weight
        self.description_weight = description_weight

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return True

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input."""
        return False

    def extract_defects_from_review(
        self, review_text: str
    ) -> list[ReferenceReviewComment]:
        """Extract defects from review text that follows the REVIEW_PROMPT format.

        Args:
            review_text: The code review text containing defects in <defect> tags

        Returns:
            List of ReferenceReviewComment objects extracted from the review text
        """
        defects = []

        # Pattern to match <defect> blocks
        defect_pattern = r"<defect>\s*(.*?)\s*</defect>"
        defect_matches = re.findall(defect_pattern, review_text, re.DOTALL)

        for defect_content in defect_matches:
            # Extract file_path, line, and suggestion from defect content
            file_path_match = re.search(r"file_path:\s*(.+)", defect_content)
            line_match = re.search(r"line:\s*(\d+)", defect_content)
            suggestion_match = re.search(
                r"suggestion:\s*(.+)", defect_content, re.DOTALL
            )

            if file_path_match and suggestion_match:
                file_path = file_path_match.group(1).strip()
                line_num = int(line_match.group(1)) if line_match else None
                suggestion = suggestion_match.group(1).strip()

                defect = ReferenceReviewComment(
                    text=suggestion,
                    path=file_path,
                    diff_hunk=None,
                    line=line_num,
                    start_line=None,
                    original_line=None,
                    original_start_line=None,
                )
                defects.append(defect)

        return defects

    def _calculate_location_similarity(
        self, pred_defect: ReferenceReviewComment, ref_defect: ReferenceReviewComment
    ) -> float:
        """Calculate location similarity between predicted and reference defects.

        Args:
            pred_defect: Predicted defect
            ref_defect: Reference defect

        Returns:
            Location similarity score between 0 and 1
        """
        # File path similarity - exact match only
        pred_path = pred_defect.path if pred_defect.path else ""
        ref_path = ref_defect.path if ref_defect.path else ""

        # Exact path match only
        if pred_path == ref_path:
            path_score = 1.0
        else:
            path_score = 0.0

        # Line number similarity
        line_score = 0.0
        if pred_defect.line is not None and ref_defect.line is not None:
            line_diff = abs(pred_defect.line - ref_defect.line)
            if line_diff == 0:
                line_score = 1.0
            elif line_diff <= 5:
                # Decay function for nearby lines
                line_score = 1.0 - (line_diff / 10.0)
            else:
                line_score = 0.1  # Small score for same file but distant lines
        elif pred_defect.line is None and ref_defect.line is None:
            line_score = 0.5  # Partial score when both don't specify line numbers

        # Combine path and line scores
        location_score = (path_score * 0.7) + (line_score * 0.3)
        return min(1.0, max(0.0, location_score))

    def _calculate_description_similarity(
        self, pred_defect: ReferenceReviewComment, ref_defect: ReferenceReviewComment
    ) -> float:
        """Calculate description similarity between predicted and reference defects.

        Args:
            pred_defect: Predicted defect
            ref_defect: Reference defect

        Returns:
            Description similarity score between 0 and 1
        """
        pred_text = pred_defect.text.lower().strip() if pred_defect.text else ""
        ref_text = ref_defect.text.lower().strip() if ref_defect.text else ""

        if not pred_text or not ref_text:
            return 0.0

        # Use difflib.SequenceMatcher for text similarity
        similarity = difflib.SequenceMatcher(None, pred_text, ref_text).ratio()
        return similarity

    def _find_best_matches(
        self,
        predicted_defects: list[ReferenceReviewComment],
        reference_defects: list[ReferenceReviewComment],
    ) -> list[dict]:
        """Find best matches between predicted and reference defects.

        Args:
            predicted_defects: List of predicted defects
            reference_defects: List of reference defects

        Returns:
            List of match results with scores
        """
        matches = []
        used_references = set()

        for pred_idx, pred_defect in enumerate(predicted_defects):
            best_match = None
            best_score = 0.0
            best_ref_idx = -1

            for ref_idx, ref_defect in enumerate(reference_defects):
                if ref_idx in used_references:
                    continue

                # Calculate combined similarity score
                location_sim = self._calculate_location_similarity(
                    pred_defect, ref_defect
                )
                description_sim = self._calculate_description_similarity(
                    pred_defect, ref_defect
                )

                combined_score = (
                    self.location_weight * location_sim
                    + self.description_weight * description_sim
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        "predicted_idx": pred_idx,
                        "reference_idx": ref_idx,
                        "location_similarity": location_sim,
                        "description_similarity": description_sim,
                        "combined_score": combined_score,
                        "predicted_defect": pred_defect,
                        "reference_defect": ref_defect,
                    }
                    best_ref_idx = ref_idx

            if (
                best_match and best_score > 0.1
            ):  # Minimum threshold for considering a match
                matches.append(best_match)
                used_references.add(best_ref_idx)
            else:
                # No good match found
                matches.append(
                    {
                        "predicted_idx": pred_idx,
                        "reference_idx": -1,
                        "location_similarity": 0.0,
                        "description_similarity": 0.0,
                        "combined_score": 0.0,
                        "predicted_defect": pred_defect,
                        "reference_defect": None,
                    }
                )

        return matches

    def _evaluate(
        self,
        *,
        prediction: CodeReviewPrediction,
        reference: list[ReferenceReviewComment],
        input: CodeReviewTaskInstance,
    ) -> dict:
        """Evaluate code review prediction against reference defects.

        Args:
            prediction: The code review prediction containing review text
            reference: List of reference review comments (defects)
            input: The input task instance (not used in this evaluator)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Extract predicted defects from review text
        predicted_defects = self.extract_defects_from_review(prediction.review_text)

        if not predicted_defects and not reference:
            # Both empty - perfect match
            return {
                "score": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "num_predicted": 0,
                "num_reference": 0,
                "matches": [],
            }

        if not predicted_defects:
            # No predictions but there are references - all misses
            return {
                "score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_predicted": 0,
                "num_reference": len(reference),
                "matches": [],
            }

        if not reference:
            # Predictions but no references - all false positives
            return {
                "score": 0.0,
                "precision": 0.0,
                "recall": 1.0,  # No reference to recall
                "f1": 0.0,
                "num_predicted": len(predicted_defects),
                "num_reference": 0,
                "matches": [],
            }

        # Find best matches between predicted and reference defects
        matches = self._find_best_matches(predicted_defects, reference)

        # Calculate metrics
        true_positives = sum(
            1 for match in matches if match["combined_score"] > 0.3
        )  # Threshold for counting as TP
        total_predicted = len(predicted_defects)
        total_reference = len(reference)

        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        recall = true_positives / total_reference if total_reference > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Overall score is the average of matched scores
        matched_scores = [match["combined_score"] for match in matches]
        average_match_score = (
            sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
        )

        # Combine metrics for final score
        score = (f1 * 0.6) + (average_match_score * 0.4)

        return {
            "score": score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_match_score": average_match_score,
            "num_predicted": total_predicted,
            "num_reference": total_reference,
            "num_true_positives": true_positives,
            "matches": [
                {
                    "predicted_idx": match["predicted_idx"],
                    "reference_idx": match["reference_idx"],
                    "location_similarity": match["location_similarity"],
                    "description_similarity": match["description_similarity"],
                    "combined_score": match["combined_score"],
                }
                for match in matches
            ],
        }

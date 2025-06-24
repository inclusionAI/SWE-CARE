import json
import re
from typing import Any

from openai import OpenAI

from swe_reason_bench.harness.evaluators import Evaluator
from swe_reason_bench.schema.dataset import (
    CodeReviewTaskInstance,
    ReferenceReviewComment,
)
from swe_reason_bench.schema.evaluation import CodeReviewPrediction

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

    llm_client: OpenAI
    llm_model: str

    def _parse_json(self, text: str) -> dict:
        # Try to find JSON string within triple backticks, assuming there are possibly multiple json markdown string
        matches = re.finditer(r"```json(.*?)```", text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.group(1))
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
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prediction.review + "\n" + EVALUATION_PROMPT,
                },
            ],
        )
        answer = response.choices[0].message.content

        return self._parse_json(answer)


class RuleBasedEvaluator(Evaluator):
    """Evaluator for code review predictions against benchmark dataset."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return True

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input."""
        return True

    def _evaluate(
        self,
        *,
        prediction: CodeReviewPrediction,
        reference: list[ReferenceReviewComment],
        input: CodeReviewTaskInstance,
    ) -> dict:
        raise NotImplementedError

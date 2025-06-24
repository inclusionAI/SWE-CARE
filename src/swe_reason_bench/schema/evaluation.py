from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CodeReviewPrediction:
    """Schema for code review prediction instances."""

    instance_id: str
    """The instance ID of the code review task"""
    review: str
    """The prediction of the code review"""
    review_trajectory: list[str] | None = None
    """The trajectory of the code review, including the intermediate steps"""


@dataclass_json
@dataclass
class EvaluatorResult:
    """Schema for evaluator result instances."""

    evaluator: str
    """The type of the evaluator"""
    evaluation: dict[str, Any]
    """The evaluation of the code review"""


@dataclass_json
@dataclass
class CodeReviewEvaluationResult:
    """Schema for code review evaluation instances."""

    instance_id: str
    """The instance ID of the code review task"""
    evaluations: list[EvaluatorResult]
    """The evaluation of the code review"""

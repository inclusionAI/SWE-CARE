"""
Run evaluation on code review predictions.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from swe_care.harness.evaluators import Evaluator
from swe_care.harness.evaluators.code_review import (
    LLMEvaluator,
    RuleBasedEvaluator,
)
from swe_care.schema.evaluation import (
    CodeReviewEvaluationResult,
    EvaluatorResult,
)
from swe_care.utils.llm_models import init_llm_client, parse_model_args
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.load import load_code_review_dataset, load_code_review_predictions


class EvaluatorType(str, Enum):
    """The types of the evaluators."""

    LLM_EVALUATOR = "llm_evaluator"
    """The LLM evaluator."""

    RULE_BASED_EVALUATOR = "rule_based_evaluator"
    """The rule-based evaluator."""


_EVALUATOR_MAP: dict[EvaluatorType, type[Evaluator]] = {
    EvaluatorType.LLM_EVALUATOR: LLMEvaluator,
    EvaluatorType.RULE_BASED_EVALUATOR: RuleBasedEvaluator,
}


def load_evaluator(
    evaluator_type: EvaluatorType,
    *,
    model_client: Optional[BaseModelClient] = None,
    **kwargs: Any,
) -> Evaluator:
    """Load the evaluator based on the type."""
    if evaluator_type not in _EVALUATOR_MAP:
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}"
            f"\nValid types are: {list(_EVALUATOR_MAP.keys())}"
        )
    evaluator_cls = _EVALUATOR_MAP[evaluator_type]
    if issubclass(evaluator_cls, LLMEvaluator):
        if model_client is None:
            raise ValueError("LLM model client is required for LLM evaluator")
        return evaluator_cls(model_client=model_client, **kwargs)
    return evaluator_cls(**kwargs)


def code_review_eval(
    dataset_file: Path | str,
    predictions_path: Path | str,
    output_dir: Path | str,
    evaluator_types: list[EvaluatorType],
    model: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_args: Optional[str] = None,
) -> None:
    """
    Run evaluation on code review predictions.

    Args:
        dataset_file: Path to the dataset file (code_review_task_instances.jsonl)
        predictions_path: Path to predictions file or directory containing predictions
        output_dir: Directory where the final_report.json will be saved
        evaluator_types: List of evaluator types to use
        model: Model name to use for LLM evaluation (required if using LLM evaluator)
        model_provider: Model provider (required if using LLM evaluator)
        model_args: Comma-separated model arguments
    """
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    if isinstance(predictions_path, str):
        predictions_path = Path(predictions_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    instances = load_code_review_dataset(dataset_file)
    predictions = load_code_review_predictions(predictions_path)

    # Initialize LLM client if needed
    model_client = None
    if EvaluatorType.LLM_EVALUATOR in evaluator_types:
        if not model or not model_provider:
            raise ValueError("Model and model provider are required for LLM evaluator")

        model_kwargs = parse_model_args(model_args)
        model_client = init_llm_client(model, model_provider, **model_kwargs)
        logger.info(f"Initialized {model_provider} client with model {model}")

    evaluators = [
        load_evaluator(
            evaluator_type,
            model_client=model_client,
        )
        for evaluator_type in evaluator_types
    ]

    all_instances_evaluation_results: list[CodeReviewEvaluationResult] = []

    for instance in tqdm(
        instances,
        desc=f"Evaluating instances with [{', '.join([e.value for e in evaluator_types])}]",
    ):
        prediction = [p for p in predictions if p.instance_id == instance.instance_id]
        if not prediction:
            logger.warning(f"No prediction found for instance {instance.instance_id}")
            continue
        prediction = prediction[0]

        evaluation_results: list[EvaluatorResult] = []
        for evaluator in evaluators:
            evaluation = None
            try:
                evaluation = evaluator.evaluate(
                    prediction=prediction,
                    reference=instance.reference_review_comments,
                    input=instance,
                )

            except Exception as e:
                logger.error(
                    f"Error evaluating instance {instance.instance_id} with {evaluator.evaluation_name}: {e}"
                )
                evaluation = {
                    "error": str(e),
                }

            evaluation_results.append(
                EvaluatorResult(
                    evaluator=evaluator.evaluation_name,
                    evaluation=evaluation,
                )
            )

        all_instances_evaluation_results.append(
            CodeReviewEvaluationResult(
                instance_id=instance.instance_id,
                evaluations=evaluation_results,
            )
        )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"{predictions_path.stem}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # Save the evaluation result
    with open(output_file, "w") as f:
        for evaluation_result in all_instances_evaluation_results:
            f.write(evaluation_result.to_json() + "\n")

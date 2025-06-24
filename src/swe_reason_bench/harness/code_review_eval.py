from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from swe_reason_bench.harness.evaluators import Evaluator
from swe_reason_bench.harness.evaluators.code_review import (
    LLMEvaluator,
    RuleBasedEvaluator,
)
from swe_reason_bench.schema.dataset import CodeReviewTaskInstance
from swe_reason_bench.schema.evaluation import (
    CodeReviewEvaluationResult,
    CodeReviewPrediction,
    EvaluatorResult,
)


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
    llm_client: Optional[OpenAI] = None,
    llm_model: Optional[str] = None,
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
        if llm_client is None:
            raise ValueError("LLM client is required for LLM evaluator")
        if llm_model is None:
            raise ValueError("LLM model is required for LLM evaluator")
        evaluator_cls.llm_client = llm_client
        evaluator_cls.llm_model = llm_model
    return evaluator_cls(**kwargs)


def load_dataset(dataset_file: Path) -> list[CodeReviewTaskInstance]:
    """Load the dataset instances from the JSONL file."""
    logger.info("Loading dataset instances...")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    dataset_instances: list[CodeReviewTaskInstance] = []

    with open(dataset_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = CodeReviewTaskInstance.from_json(line.strip())
                dataset_instances.append(instance)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e

    logger.success(f"Loaded {len(dataset_instances)} dataset instances")
    return dataset_instances


def load_predictions(predictions_path: Path) -> list[CodeReviewPrediction]:
    """Load the predictions from the JSONL file."""
    logger.info("Loading predictions...")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions: list[CodeReviewPrediction] = []

    with open(predictions_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                prediction = CodeReviewPrediction.from_json(line.strip())
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e
    logger.success(f"Loaded {len(predictions)} predictions")
    return predictions


def code_review_eval(
    dataset_file: Path,
    predictions_path: Path,
    output_dir: Path,
    evaluator_types: list[EvaluatorType],
    llm_client: Optional[OpenAI] = None,
    llm_model: Optional[str] = None,
) -> None:
    """
    Run evaluation on code review predictions.

    Args:
        dataset_file: Path to the dataset file (code_review_task_instances.jsonl)
        predictions_path: Path to predictions file or directory containing predictions
        output_dir: Directory where the final_report.json will be saved
    """
    instances = load_dataset(dataset_file)
    predictions = load_predictions(predictions_path)

    evaluators = [
        load_evaluator(
            evaluator_type,
            llm_client=llm_client,
            llm_model=llm_model,
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
    output_file = output_dir / "final_report.jsonl"

    # Save the evaluation result
    with open(output_file, "w") as f:
        for evaluation_result in all_instances_evaluation_results:
            f.write(evaluation_result.to_json() + "\n")

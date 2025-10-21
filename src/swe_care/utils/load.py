from pathlib import Path

from datasets import load_dataset
from loguru import logger

from swe_care.schema.dataset import CodeReviewTaskInstance
from swe_care.schema.evaluation import CodeReviewEvaluationResult, CodeReviewPrediction
from swe_care.schema.inference import CodeReviewInferenceInstance


def load_code_review_dataset(
    dataset_name_or_path: Path | str = "inclusionAI/SWE-CARE",
) -> list[CodeReviewTaskInstance]:
    """Load the code review dataset instances from a JSONL file or Hugging Face dataset.

    Args:
        dataset_name_or_path: Either a file path to a local JSONL file, or a Hugging Face dataset name.
                             Defaults to "inclusionAI/SWE-CARE".

    Returns:
        List of CodeReviewTaskInstance objects

    Raises:
        FileNotFoundError: If a file path is provided but the file doesn't exist
        Exception: If there's an error parsing the file or loading from Hugging Face
    """
    # Check if dataset_name_or_path is a file path
    if isinstance(dataset_name_or_path, str):
        path = Path(dataset_name_or_path)
    else:
        path = dataset_name_or_path

    # If it's an existing file, load from file
    if path.exists():
        logger.info("Loading dataset instances from file...")

        dataset_instances: list[CodeReviewTaskInstance] = []

        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    instance = CodeReviewTaskInstance.from_json(line.strip())
                    dataset_instances.append(instance)
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    raise e

        logger.success(f"Loaded {len(dataset_instances)} dataset instances from file")
        return dataset_instances

    # Otherwise, load from Hugging Face
    else:
        logger.info(f"Loading dataset from Hugging Face: {dataset_name_or_path}")

        # Load the test split from Hugging Face
        dataset = load_dataset(
            str(dataset_name_or_path), split="test", revision="0.2.0"
        )

        # Convert Hugging Face dataset to list of CodeReviewTaskInstance
        dataset_instances: list[CodeReviewTaskInstance] = []

        logger.info("Processing test split")
        for idx, item in enumerate(dataset):
            try:
                # Convert the Hugging Face dataset item to CodeReviewTaskInstance
                # Using from_dict method provided by dataclass_json
                instance = CodeReviewTaskInstance.from_dict(item)
                dataset_instances.append(instance)
            except Exception as e:
                logger.error(f"Error converting item {idx} in test split: {e}")
                raise e

        logger.success(
            f"Loaded {len(dataset_instances)} dataset instances from Hugging Face test split"
        )
        return dataset_instances


def load_code_review_predictions(
    predictions_path: Path | str,
) -> list[CodeReviewPrediction]:
    """Load the code review predictions from the JSONL file."""
    if isinstance(predictions_path, str):
        predictions_path = Path(predictions_path)
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


def load_code_review_text(
    dataset_file: Path | str,
) -> list[CodeReviewInferenceInstance]:
    """Load the code review text dataset instances from the JSONL file.

    Args:
        dataset_file: Path to the input JSONL file containing CodeReviewInferenceInstance objects

    Returns:
        List of CodeReviewInferenceInstance objects

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        Exception: If there's an error parsing the file
    """
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    logger.info("Loading inference text dataset instances...")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    dataset_instances: list[CodeReviewInferenceInstance] = []

    with open(dataset_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = CodeReviewInferenceInstance.from_json(line.strip())
                dataset_instances.append(instance)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e

    logger.success(f"Loaded {len(dataset_instances)} inference text dataset instances")
    return dataset_instances


def load_code_review_eval_result(
    eval_result_file: Path | str,
) -> list[CodeReviewEvaluationResult]:
    """Load the code review evaluation results from the JSONL file.

    Args:
        eval_result_file: Path to the evaluation result JSONL file

    Returns:
        List of CodeReviewEvaluationResult objects

    Raises:
        FileNotFoundError: If the evaluation result file doesn't exist
        Exception: If there's an error parsing the file
    """
    if isinstance(eval_result_file, str):
        eval_result_file = Path(eval_result_file)
    logger.info(f"Loading evaluation results from {eval_result_file}...")

    if not eval_result_file.exists():
        raise FileNotFoundError(f"Evaluation result file not found: {eval_result_file}")

    eval_results: list[CodeReviewEvaluationResult] = []

    with open(eval_result_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = CodeReviewEvaluationResult.from_json(line.strip())
                eval_results.append(result)
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                raise e

    logger.success(f"Loaded {len(eval_results)} evaluation results")
    return eval_results

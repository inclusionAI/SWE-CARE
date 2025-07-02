"""
Run inference on a code review dataset using LLM APIs.

This module provides functionality to run inference on code review datasets using
OpenAI or Anthropic APIs with multithreading support and progress tracking.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from swe_care.schema.evaluation import CodeReviewPrediction
from swe_care.schema.inference import CodeReviewInferenceInstance
from swe_care.utils.llm_models import init_llm_client, parse_model_args
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.load import load_code_review_text


def run_api_instance(
    model_client: BaseModelClient,
    instance: CodeReviewInferenceInstance,
) -> CodeReviewPrediction:
    """Process a single instance and return the prediction."""
    system_messages = instance.text.split("\n", 1)[0]
    user_message = instance.text.split("\n", 1)[1]
    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_messages},
        {"role": "user", "content": user_message},
    ]

    # Get completion
    completion = model_client.create_completion(messages)

    # Create prediction
    prediction = CodeReviewPrediction(
        instance_id=instance.instance_id,
        review_text=completion,
    )

    return prediction


def run_api(
    dataset_file: Path | str,
    model: str,
    model_provider: str,
    model_args: str | None,
    output_dir: Path | str,
    jobs: int = 2,
    skip_existing: bool = True,
) -> None:
    """
    Run inference on a code review dataset using LLM APIs.

    Args:
        dataset_file: Path to the input dataset file containing CodeReviewInferenceInstance objects
        model: Model name to use for inference
        model_provider: Model provider (openai, anthropic)
        model_args: Comma-separated model arguments (e.g., 'top_p=0.95,temperature=0.70')
        output_dir: Directory to save the generated predictions
        jobs: Number of parallel jobs for inference
        skip_existing: Whether to skip existing predictions in the output file
    """
    logger.info(
        f"Starting inference with model={model}, provider={model_provider}, jobs={jobs}"
    )

    # Convert paths to Path objects
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Validate input file
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model arguments
    model_kwargs = parse_model_args(model_args)
    logger.info(f"Using model arguments: {model_kwargs}")

    # Initialize LLM client
    model_client = init_llm_client(model, model_provider, **model_kwargs)
    logger.info(f"Successfully initialized {model_provider} client with model {model}")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_file}")
    instances = load_code_review_text(dataset_file)

    # Define output file
    output_file = output_dir / f"{dataset_file.stem}__{model}.jsonl"
    logger.info(f"Output will be saved to {output_file}")

    # Load existing predictions if skip_existing is True
    existing_predictions = set()
    if skip_existing and output_file.exists():
        logger.info("Loading existing predictions to skip...")
        try:
            with open(output_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pred = CodeReviewPrediction.from_json(line)
                            existing_predictions.add(pred.instance_id)
                        except Exception as e:
                            logger.warning(f"Could not parse existing prediction: {e}")
            logger.info(f"Found {len(existing_predictions)} existing predictions")
        except Exception as e:
            logger.warning(f"Could not load existing predictions: {e}")

    # Filter out instances that already have predictions
    if skip_existing:
        instances_to_process = [
            instance
            for instance in instances
            if instance.instance_id not in existing_predictions
        ]
        logger.info(
            f"Skipping {len(instances) - len(instances_to_process)} instances with existing predictions"
        )
    else:
        instances_to_process = instances
        if output_file.exists():
            output_file.unlink()

    if not instances_to_process:
        logger.info("No instances to process. All instances already have predictions.")
        return

    # Sort instances by text length (ascending) for better load balancing
    logger.info("Sorting instances by text length for better load balancing...")
    instances_to_process.sort(key=lambda x: len(x.text))

    # Set up thread-safe writing
    write_lock = threading.Lock()
    # Run inference with multithreading
    successful_predictions = 0
    failed_predictions = 0

    logger.info(
        f"Starting inference on {len(instances_to_process)} instances with {jobs} threads..."
    )

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(run_api_instance, model_client, instance): instance
            for instance in instances_to_process
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(instances_to_process), desc="Processing instances") as pbar:
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]

                try:
                    prediction = future.result()
                    with write_lock:
                        with open(output_file, "a") as f:
                            f.write(prediction.to_json() + "\n")
                    successful_predictions += 1

                except Exception as e:
                    failed_predictions += 1
                    logger.error(
                        f"Exception processing instance {instance.instance_id}: {e}"
                    )

                pbar.update(1)
                pbar.set_postfix(
                    {"success": successful_predictions, "failed": failed_predictions}
                )

    # Final summary
    total_processed = successful_predictions + failed_predictions
    logger.info(
        f"Inference completed! Processed {total_processed} instances: "
        f"{successful_predictions} successful, {failed_predictions} failed"
    )
    logger.info(f"Predictions saved to {output_file}")

    if failed_predictions > 0:
        logger.warning(
            f"{failed_predictions} instances failed processing. Check logs for details."
        )

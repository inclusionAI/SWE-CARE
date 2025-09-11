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
from swe_care.utils.llm_models import get_model_info, init_llm_client, parse_model_args
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.load import load_code_review_predictions, load_code_review_text

# Import API-specific exceptions
try:
    from openai import BadRequestError as OpenAIBadRequestError
except ImportError:
    OpenAIBadRequestError = None

try:
    from anthropic import BadRequestError as AnthropicBadRequestError
except ImportError:
    AnthropicBadRequestError = None


def sanitize_filename(name: str) -> str:
    """Sanitize model name for use in filenames."""
    # Replace forward slashes and other problematic characters
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def reduce_text_to_fit_context(
    text: str,
    model_client: BaseModelClient,
    max_input_tokens: int,
    buffer_tokens: int = 500,
) -> str:
    """Reduce text to fit within model's context window.

    Args:
        text: The full text to potentially reduce
        model_client: The LLM client to use for token counting
        max_input_tokens: Maximum input tokens for the model
        buffer_tokens: Buffer to leave for completion (default 500)

    Returns:
        Reduced text that fits within context window

    Raises:
        ValueError: If text cannot be reduced to fit context window
    """
    # Split text into system and user messages
    parts = text.split("\n", 1)
    if len(parts) != 2:
        raise ValueError(
            "Text must contain system and user messages separated by newline"
        )

    system_message = parts[0]
    user_message = parts[1]

    # Prepare messages for token counting
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Count initial tokens
    current_tokens = model_client.count_tokens_from_messages(messages)
    target_tokens = max_input_tokens - buffer_tokens

    if current_tokens <= target_tokens:
        return text  # No reduction needed

    logger.info(
        f"Text exceeds context window ({current_tokens} > {target_tokens}). "
        "Attempting to reduce..."
    )

    # Find code block if present
    code_start = user_message.find("<code>")
    code_end = user_message.find("</code>")

    if code_start == -1 or code_end == -1:
        # No code block found - return text as-is
        logger.warning(
            f"Text exceeds context window ({current_tokens} tokens) but contains no <code> block to truncate. "
            "Returning text as-is."
        )
        return text

    # Extract parts
    before_code = user_message[:code_start]
    code_block = user_message[code_start + 6 : code_end]  # Skip "<code>"
    after_code = user_message[code_end + 7 :]  # Skip "</code>"

    # Parse file sections within code block
    file_sections = []
    current_pos = 0

    while True:
        start_marker = "[start of "
        end_marker = "[end of "

        start_idx = code_block.find(start_marker, current_pos)
        if start_idx == -1:
            break

        # Find file path
        path_end = code_block.find("]", start_idx)
        if path_end == -1:
            break

        file_path = code_block[start_idx + len(start_marker) : path_end]

        # Find corresponding end marker
        end_pattern = f"{end_marker}{file_path}]"
        end_idx = code_block.find(end_pattern, path_end)
        if end_idx == -1:
            break

        # Extract file content
        content_start = path_end + 2  # Skip "]\n"
        file_content = code_block[content_start:end_idx]

        file_sections.append(
            {
                "path": file_path,
                "start": start_idx,
                "end": end_idx + len(end_pattern),
                "content": file_content,
                "content_start": content_start,
                "content_end": end_idx,
            }
        )

        current_pos = end_idx + len(end_pattern)

    if not file_sections:
        # No file sections found, try to remove entire code block
        new_user_message = before_code + after_code
        new_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": new_user_message},
        ]
        new_tokens = model_client.count_tokens_from_messages(new_messages)

        if new_tokens > target_tokens:
            raise ValueError(
                f"Text is still too long ({new_tokens} tokens) after removing code block"
            )

        logger.info("Removed entire <code> block to fit context")
        return system_message + "\n" + new_user_message

    # Truncate file contents from right to left
    max_truncation_retries = 10
    truncation_attempts = 0
    while (
        current_tokens > target_tokens
        and file_sections
        and truncation_attempts < max_truncation_retries
    ):
        # Get the rightmost file section
        last_file = file_sections[-1]

        # Try truncating the content by 20%
        content = last_file["content"]
        if len(content) > 100:
            # Truncate from the right
            new_length = int(len(content) * 0.8)
            last_file["content"] = content[:new_length] + "\n... (content truncated)"
        else:
            # Remove this file section entirely if it's too small to truncate further
            file_sections.pop()

        # Rebuild the entire code block from the updated file sections
        new_code_block = ""
        for section in file_sections:
            new_code_block += f"[start of {section['path']}]\n"
            new_code_block += section["content"]
            if not section["content"].endswith("\n"):
                new_code_block += "\n"
            new_code_block += f"[end of {section['path']}]"
            if section != file_sections[-1]:  # Add newline between sections
                new_code_block += "\n"

        # Update code_block
        code_block = new_code_block

        # Recount tokens
        if code_block.strip():
            new_user_message = (
                before_code + "<code>" + code_block + "</code>" + after_code
            )
        else:
            new_user_message = before_code + after_code

        new_messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": new_user_message},
        ]
        current_tokens = model_client.count_tokens_from_messages(new_messages)
        truncation_attempts += 1

        logger.debug(f"After truncation: {current_tokens} tokens")

    if current_tokens > target_tokens:
        if truncation_attempts >= max_truncation_retries:
            # Still try
            return system_message + "\n" + new_user_message
        raise ValueError(
            f"Unable to reduce text to fit context window. "
            f"Current: {current_tokens}, Target: {target_tokens}"
        )

    logger.info(f"Successfully reduced text to {current_tokens} tokens")
    return system_message + "\n" + new_user_message


def run_api_instance(
    model_client: BaseModelClient,
    instance: CodeReviewInferenceInstance,
) -> CodeReviewPrediction | None:
    """Process a single instance and return the prediction with automatic retry on token limit errors.

    This function implements a robust retry mechanism to handle token counting inaccuracies
    across different model providers. Since token counting using tiktoken is approximate
    for non-OpenAI models, the function automatically retries with progressively smaller
    context windows when encountering token limit errors.

    Retry Logic:
    1. Starts with the full context window size from the model configuration
    2. If a BadRequestError related to token limits is encountered, reduces the max tokens by 20%
    3. Continues retrying until either:
       - The request succeeds
       - The max tokens falls below 50% of the original limit (returns None)

    The function handles various types of token limit errors:
    - OpenAI BadRequestError exceptions
    - Anthropic BadRequestError exceptions
    - Generic errors containing keywords: "bad request", "context length", "token",
      "too long", "invalid_parameter_error"

    Args:
        model_client: The LLM client to use for inference and token counting
        instance: The code review inference instance containing the text to process

    Returns:
        CodeReviewPrediction: Successfully generated prediction
        None: If any of the following occurs:
            - Text cannot be reduced to fit even 50% of the original context window
            - ValueError is raised during text reduction (e.g., malformed input)
            - Token limit errors persist after reducing to 50% threshold

    Raises:
        Exception: Re-raises any exceptions that are not token limit related
    """
    try:
        # Get model info for token limits
        _, original_max_input_tokens = get_model_info(model_client.model)

        # Start with the full context window
        current_max_tokens = original_max_input_tokens
        min_max_tokens = (
            original_max_input_tokens * 0.5
        )  # 50% threshold until we give up

        # Retry with progressively smaller context windows
        max_reduction_retries = 10
        reduction_attempts = 0
        while current_max_tokens >= min_max_tokens:
            if reduction_attempts >= max_reduction_retries:
                logger.error(
                    f"Failed to process instance {instance.instance_id}: exceeded max retries (10) for context reduction"
                )
                return None
            try:
                # Try to reduce text if needed
                reduced_text = reduce_text_to_fit_context(
                    instance.text, model_client, current_max_tokens
                )

                # Split reduced text into system and user messages
                system_message, user_message = reduced_text.split("\n", 1)

                # Prepare messages for the LLM
                messages = [
                    {"role": "system", "content": system_message},
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

            except Exception as e:
                # Check if it's a BadRequestError related to token limits
                is_bad_request = False
                error_message = str(e).lower()

                # Check for OpenAI BadRequestError
                if OpenAIBadRequestError and isinstance(e, OpenAIBadRequestError):
                    is_bad_request = True
                # Check for Anthropic BadRequestError
                elif AnthropicBadRequestError and isinstance(
                    e, AnthropicBadRequestError
                ):
                    is_bad_request = True
                # Check for generic BadRequestError patterns in error message
                elif (
                    "bad request" in error_message
                    or "token too long" in error_message
                    or "invalid_parameter_error" in error_message
                ):
                    is_bad_request = True

                if is_bad_request:
                    # Reduce max tokens by 20% and retry
                    new_max_tokens = int(current_max_tokens * 0.8)
                    logger.warning(
                        f"BadRequest error for instance {instance.instance_id}. "
                        f"Reducing max tokens from {current_max_tokens} to {new_max_tokens}"
                    )
                    current_max_tokens = new_max_tokens
                    reduction_attempts += 1

                    if current_max_tokens < min_max_tokens:
                        logger.error(
                            f"Failed to process instance {instance.instance_id}: "
                            f"Context window reduced below 50% threshold ({min_max_tokens} tokens)"
                        )
                        return None
                else:
                    # Not a token limit issue, re-raise
                    raise

    except ValueError as e:
        logger.error(
            f"Failed to process instance {instance.instance_id} due to context window: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error processing instance {instance.instance_id}: {e}"
        )
        raise


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

    # Define output file (sanitize model name for filesystem)
    safe_model_name = sanitize_filename(model)
    output_file = output_dir / f"{dataset_file.stem}__{safe_model_name}.jsonl"
    logger.info(f"Output will be saved to {output_file}")

    # Load existing predictions if skip_existing is True
    existing_prediction_ids = None
    if skip_existing and output_file.exists():
        logger.info(f"Loading existing predictions from {output_file} to skip...")
        try:
            existing_predictions = load_code_review_predictions(output_file)
            existing_prediction_ids = {
                prediction.instance_id for prediction in existing_predictions
            }
            logger.info(f"Found {len(existing_predictions)} existing predictions")
        except Exception as e:
            logger.warning(f"Could not load existing predictions: {e}")

    # Filter out instances that already have predictions
    if skip_existing and existing_prediction_ids is not None:
        instances_to_process = [
            instance
            for instance in instances
            if instance.instance_id not in existing_prediction_ids
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
                    if prediction is not None:
                        with write_lock:
                            with open(output_file, "a") as f:
                                f.write(prediction.to_json() + "\n")
                        successful_predictions += 1
                    else:
                        # Prediction was None due to context window issues
                        failed_predictions += 1

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

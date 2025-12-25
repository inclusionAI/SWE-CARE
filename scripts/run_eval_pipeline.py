#!/usr/bin/env python3
"""
Bootstrap script for running the complete SWE-CARE evaluation pipeline.

This script automates the following steps:
1. Generate text datasets from collected SWE-CARE data
2. Run LLM inference on code review tasks
3. Evaluate model predictions using LLM evaluator (default: OpenAI o3)
"""

import argparse
import glob
import json
import os
import runpy
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from swe_care.inference.run_api import sanitize_filename
from swe_care.utils.llm_models import get_available_models_and_providers


def setup_logging(output_dir: Path, timestamp: str):
    """Set up loguru logger to write to both console and file."""
    log_file = output_dir / f"pipeline_{timestamp}.log"
    logger.add(log_file, rotation="500 MB", retention="10 days")
    logger.info(f"Logging to {log_file}")


def validate_environment():
    """Validate required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for model inference",
        "LLM_EVALUATOR_OPENAI_API_KEY": "OpenAI API key for LLM evaluation",
    }

    missing_vars = []
    for var, desc in required_vars.items():
        if not os.environ.get(var):
            missing_vars.append(f"{var} ({desc})")

    if missing_vars:
        logger.error("Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"  - {var}")
        sys.exit(1)

    # Log optional environment variables
    optional_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key (if using Claude models)",
        "OPENAI_BASE_URL": "Custom OpenAI-compatible API endpoint",
        "ANTHROPIC_BASE_URL": "Custom Anthropic-compatible API endpoint",
        "LLM_EVALUATOR_OPENAI_BASE_URL": "Custom OpenAI-compatible API endpoint for evaluation",
    }

    for var, desc in optional_vars.items():
        if os.environ.get(var):
            logger.info(f"✓ {var} is set ({desc})")


def get_llm_evaluator_model_args(evaluator_model: str) -> str:
    evaluator_model_normalized = evaluator_model.strip()
    if evaluator_model_normalized.lower() == "o3":
        return "temperature=1"
    return "temperature=0"


def save_pipeline_config(output_dir: Path, config: dict, timestamp: str):
    """Save pipeline configuration to JSON file."""
    config_file = output_dir / f"pipeline_config_{timestamp}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved pipeline configuration to {config_file}")
    return config_file


def run_create_code_review_text(
    dataset_name_or_path: Path | str,
    output_dir: Path,
    file_source: str,
    k: int | None,
    retrieval_output_dir: Path | None,
    tokens: list[str] | None,
    jobs: int,
    skip_existing: bool,
    use_skeleton: bool,
) -> Path:
    """Run the create_code_review_text module."""
    logger.info("=" * 80)
    logger.info("Step 1: Generating text dataset")
    logger.info("=" * 80)

    text_output_dir = output_dir / "code_review_text"

    # Build command arguments
    args = [
        "swe_care.inference",
        "create_code_review_text",
        "--dataset-name-or-path",
        str(dataset_name_or_path),
        "--output-dir",
        str(text_output_dir),
        "--file-source",
        file_source,
        "--jobs",
        str(jobs),
    ]
    if use_skeleton:
        args.append("--use-skeleton")

    if k is not None:
        args.extend(["--k", str(k)])

    if retrieval_output_dir is not None:
        args.extend(["--retrieval-output-dir", str(retrieval_output_dir)])

    if tokens:
        args.extend(["--tokens"] + tokens)

    if skip_existing:
        args.append("--skip-existing")

    logger.info(f"Running command: {' '.join(args)}")

    # Run the module
    sys.argv = args
    runpy.run_module("swe_care.inference", run_name="__main__")

    # Determine output file name based on file_source
    # Based on create_code_review_text.py
    skeleton_suffix = "__skeleton" if use_skeleton else ""

    # Determine dataset name for filename
    if isinstance(dataset_name_or_path, str):
        path = Path(dataset_name_or_path)
    else:
        path = dataset_name_or_path

    if path.exists():
        dataset_name = path.stem
    else:
        # Use the last part of the Hugging Face dataset name
        dataset_name = str(dataset_name_or_path).split("/")[-1]

    if k is not None and file_source in ["bm25", "all"]:
        output_filename = f"{dataset_name}__{file_source}__k{k}{skeleton_suffix}.jsonl"
    else:
        output_filename = f"{dataset_name}__{file_source}{skeleton_suffix}.jsonl"
    output_file = text_output_dir / output_filename

    if not output_file.exists():
        logger.error(f"Expected output file not found: {output_file}")
        sys.exit(1)

    logger.success(f"Text dataset generated: {output_file}")
    return output_file


def run_inference(
    dataset_file: Path,
    output_dir: Path,
    model: str,
    model_provider: str,
    model_args: str | None,
    jobs: int,
    skip_existing: bool,
) -> Path:
    """Run LLM inference on the dataset."""
    logger.info("=" * 80)
    logger.info("Step 2: Running LLM inference")
    logger.info("=" * 80)

    # Create model-specific subdirectory for predictions
    safe_model_name = sanitize_filename(model)
    predictions_output_dir = output_dir / "predictions" / safe_model_name

    # Build command arguments
    args = [
        "swe_care.inference",
        "run_api",
        "--dataset-file",
        str(dataset_file),
        "--output-dir",
        str(predictions_output_dir),
        "--model",
        model,
        "--model-provider",
        model_provider,
        "--jobs",
        str(jobs),
    ]

    if model_args:
        args.extend(["--model-args", model_args])

    if skip_existing:
        args.append("--skip-existing")

    logger.info(f"Running command: {' '.join(args)}")

    # Run the module
    sys.argv = args
    runpy.run_module("swe_care.inference", run_name="__main__")

    # Determine output file name (based on run_api.py)
    output_filename = f"{dataset_file.stem}__{safe_model_name}.jsonl"
    output_file = predictions_output_dir / output_filename

    if not output_file.exists():
        logger.error(f"Expected output file not found: {output_file}")
        sys.exit(1)

    logger.success(f"Predictions generated: {output_file}")
    return output_file


def run_evaluation(
    dataset_name_or_path: Path | str,
    predictions_file: Path,
    output_dir: Path,
    evaluator_model: str,
    jobs: int,
    eval_start_time: datetime,
    safe_model_name: str,
) -> Path:
    """Run evaluation using LLM evaluator."""
    logger.info("=" * 80)
    logger.info("Step 3: Running LLM evaluation")
    logger.info("=" * 80)

    # Create model-specific subdirectory for evaluation results
    evaluation_output_dir = output_dir / "evaluation" / safe_model_name

    # Set up environment for LLM evaluator
    # Save original environment variables to restore later
    original_openai_api_key = os.environ.get("OPENAI_API_KEY")
    original_openai_base_url = os.environ.get("OPENAI_BASE_URL")

    # Set evaluator-specific environment variables
    if os.environ.get("LLM_EVALUATOR_OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_EVALUATOR_OPENAI_API_KEY"]

    if os.environ.get("LLM_EVALUATOR_OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["LLM_EVALUATOR_OPENAI_BASE_URL"]
    else:
        # If no specific evaluator base URL is set, remove any existing base URL
        # to ensure the evaluator uses the default OpenAI endpoint
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

    # Build command arguments
    evaluator_model_args = get_llm_evaluator_model_args(evaluator_model)
    args = [
        "swe_care.harness",
        "code_review_eval",
        "--dataset-name-or-path",
        str(dataset_name_or_path),
        "--predictions-path",
        str(predictions_file),
        "--output-dir",
        str(evaluation_output_dir),
        "--evaluator",
        "rule_based_evaluator",
        "llm_evaluator",
        "--model",
        evaluator_model,
        "--model-provider",
        "openai",
        "--model-args",
        evaluator_model_args,
        "--jobs",
        str(jobs),
    ]

    logger.info(f"Running command: {' '.join(args)}")
    logger.info(
        f"Using {evaluator_model} model for LLM evaluation with {evaluator_model_args}"
    )

    try:
        # Run the module
        sys.argv = args
        runpy.run_module("swe_care.harness", run_name="__main__")

        # Check for output (based on code_review_eval.py)
        # The evaluation creates a file with pattern: {predictions_path.stem}_report_{timestamp}.jsonl
        report_pattern = str(
            evaluation_output_dir / f"{predictions_file.stem}_report_*.jsonl"
        )
        report_files = glob.glob(report_pattern)

        if not report_files:
            logger.error(
                f"No evaluation report found matching pattern: {report_pattern}"
            )
            raise RuntimeError("No evaluation report found")

        # Filter report files to only those created after evaluation start time
        valid_report_files = []
        for report_file_path in report_files:
            report_file = Path(report_file_path)
            # Check file modification time
            file_mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
            if file_mtime >= eval_start_time:
                valid_report_files.append(report_file_path)

        if not valid_report_files:
            logger.error(
                f"No evaluation reports found created after {eval_start_time.strftime('%Y-%m-%d %H:%M:%S')}. "
                f"Found {len(report_files)} older reports."
            )
            raise RuntimeError("No valid evaluation reports found")

        # Get the most recent valid report file
        final_report = Path(max(valid_report_files))
        logger.success(f"Evaluation complete: {final_report}")
        return final_report

    finally:
        # Restore original environment variables (always executed)
        if original_openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = original_openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if original_openai_base_url is not None:
            os.environ["OPENAI_BASE_URL"] = original_openai_base_url
        elif "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap script for running the complete SWE-CARE evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  Required:
    OPENAI_API_KEY                  - OpenAI API key for model inference
    LLM_EVALUATOR_OPENAI_API_KEY    - OpenAI API key for LLM evaluation
  
  Optional:
    ANTHROPIC_API_KEY               - Anthropic API key (if using Claude models)
    OPENAI_BASE_URL                 - Custom OpenAI-compatible API endpoint
    ANTHROPIC_BASE_URL              - Custom Anthropic-compatible API endpoint
    LLM_EVALUATOR_OPENAI_BASE_URL   - Custom OpenAI-compatible API endpoint for evaluation

Example usage:
  # Basic usage with no file context (using default Hugging Face dataset)
  python scripts/run_eval_pipeline.py \\
    --output-dir results/pipeline_output \\
    --model gpt-4o \\
    --model-provider openai \\
    --file-source none

  # With local dataset file
  python scripts/run_eval_pipeline.py \\
    --dataset-name-or-path results/dataset/code_review_task_instances.jsonl \\
    --output-dir results/pipeline_output \\
    --model gpt-4o \\
    --model-provider openai \\
    --file-source none

  # With oracle file source and custom model args
  python scripts/run_eval_pipeline.py \\
    --dataset-name-or-path results/dataset/code_review_task_instances.jsonl \\
    --output-dir results/pipeline_output \\
    --model claude-3-5-sonnet-20241022 \\
    --model-provider anthropic \\
    --model-args "temperature=0.5,max_tokens=4096" \\
    --file-source oracle \\
    --github-tokens "token1" "token2"

  # With BM25 retrieval
  python scripts/run_eval_pipeline.py \\
    --dataset-name-or-path results/dataset/code_review_task_instances.jsonl \\
    --output-dir results/pipeline_output \\
    --model "models/gemini-2.5-pro" \\
    --model-provider openai \\
    --file-source bm25 \\
    --k 10 \\
    --retrieval-output-dir results/retrieval_output
""",
    )

    # Required arguments
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        required=False,
        default="inclusionAI/SWE-CARE",
        help="Path to the input SWE-CARE dataset file or Hugging Face dataset name (default: inclusionAI/SWE-CARE)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save all pipeline outputs",
    )

    available_providers, available_models = get_available_models_and_providers()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model name to use for inference. Available models: {', '.join(available_models)}",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        required=True,
        choices=available_providers,
        default="openai" if "openai" in available_providers else None,
        help=f"Model provider. Available providers: {', '.join(available_providers)}",
    )
    parser.add_argument(
        "--model-args",
        type=str,
        required=False,
        default="temperature=0.6,top_p=0.95",
        help="List of model arguments separated by commas (e.g., 'top_p=0.95,temperature=0.70')",
    )
    parser.add_argument(
        "--evaluator-model",
        type=str,
        required=False,
        default="o3",
        help="Model name to use for LLM evaluation (OpenAI; default: o3)",
    )
    parser.add_argument(
        "--file-source",
        type=str,
        choices=["none", "oracle", "bm25", "all"],
        default="none",
        help="Source strategy for files (default: none)",
    )
    parser.add_argument(
        "--use-skeleton",
        action="store_true",
        help="Use TreeSitter-based stubs for Python files in retrieval/context",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Maximum number of files to use (required when file-source is 'bm25' or 'all')",
    )
    parser.add_argument(
        "--retrieval-output-dir",
        type=Path,
        help="Output directory for retrieval operations (required when file-source is 'bm25' or 'all')",
    )
    parser.add_argument(
        "--github-tokens",
        type=str,
        nargs="*",
        help="GitHub API token(s) for fetching data",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of parallel jobs for processing (default: 2)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances that already have predictions",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.file_source in ["bm25", "all"]:
        if args.k is None:
            parser.error(f"--k is required when --file-source is '{args.file_source}'")
        if args.retrieval_output_dir is None:
            parser.error(
                f"--retrieval-output-dir is required when --file-source is '{args.file_source}'"
            )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this pipeline run
    pipeline_start_time = datetime.now()
    timestamp_str = pipeline_start_time.strftime("%Y%m%d_%H%M%S")

    # Set up logging
    setup_logging(args.output_dir, timestamp_str)

    logger.info("Starting SWE-CARE evaluation pipeline")
    logger.info(f"Pipeline timestamp: {timestamp_str}")
    logger.info(f"Dataset: {args.dataset_name_or_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model} ({args.model_provider})")
    logger.info(f"Evaluator model: {args.evaluator_model} (openai)")
    logger.info(f"File source: {args.file_source}")

    # Validate environment
    validate_environment()

    # Save configuration
    evaluator_model_args = get_llm_evaluator_model_args(args.evaluator_model)
    config = {
        "timestamp": pipeline_start_time.isoformat(),
        "dataset_name_or_path": str(args.dataset_name_or_path),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "model_provider": args.model_provider,
        "model_args": args.model_args,
        "file_source": args.file_source,
        "k": args.k,
        "retrieval_output_dir": str(args.retrieval_output_dir)
        if args.retrieval_output_dir
        else None,
        "github_tokens_count": len(args.github_tokens) if args.github_tokens else 0,
        "jobs": args.jobs,
        "skip_existing": args.skip_existing,
        "llm_evaluator": {
            "model": args.evaluator_model,
            "model_provider": "openai",
            "model_args": evaluator_model_args,
        },
    }
    config_file = save_pipeline_config(args.output_dir, config, timestamp_str)

    try:
        # Step 1: Generate text dataset
        text_dataset_file = run_create_code_review_text(
            dataset_name_or_path=args.dataset_name_or_path,
            output_dir=args.output_dir,
            file_source=args.file_source,
            k=args.k,
            retrieval_output_dir=args.retrieval_output_dir,
            tokens=args.github_tokens,
            jobs=args.jobs,
            skip_existing=args.skip_existing,
            use_skeleton=args.use_skeleton,
        )

        # Step 2: Run inference
        predictions_file = run_inference(
            dataset_file=text_dataset_file,
            output_dir=args.output_dir,
            model=args.model,
            model_provider=args.model_provider,
            model_args=args.model_args,
            jobs=args.jobs,
            skip_existing=args.skip_existing,
        )

        # Step 3: Run evaluation
        eval_start_time = datetime.now()
        safe_model_name = sanitize_filename(args.model)
        evaluation_file = run_evaluation(
            dataset_name_or_path=args.dataset_name_or_path,
            predictions_file=predictions_file,
            output_dir=args.output_dir,
            evaluator_model=args.evaluator_model,
            jobs=args.jobs,
            eval_start_time=eval_start_time,
            safe_model_name=safe_model_name,
        )

        logger.success("=" * 80)
        logger.success("Pipeline completed successfully!")
        logger.success("=" * 80)
        logger.info(f"Text dataset: {text_dataset_file}")
        logger.info(f"Predictions: {predictions_file}")
        logger.info(f"Evaluation: {evaluation_file}")
        logger.info(f"Configuration: {config_file}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

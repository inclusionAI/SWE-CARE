import argparse
import sys
from pathlib import Path

from loguru import logger

import swe_care.harness.code_review_eval
from swe_care.harness.code_review_eval import EvaluatorType, code_review_eval
from swe_care.utils.llm_models import get_available_models_and_providers

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "code_review_eval": {
        "function": code_review_eval,
        "help": swe_care.harness.code_review_eval.__doc__,
    },
}


def create_global_parser():
    """Create a parser with global arguments that can be used as a parent parser."""
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory",
    )
    return global_parser


def get_args():
    # Parse command line manually to handle flexible argument order
    args = sys.argv[1:]

    # Find the subcommand
    subcommands = list(SUBCOMMAND_MAP.keys())
    subcommand = None
    subcommand_index = None

    for i, arg in enumerate(args):
        if arg in subcommands:
            subcommand = arg
            subcommand_index = i
            break

    # Create global parser
    global_parser = create_global_parser()

    if subcommand is None:
        # No subcommand found, use normal argparse
        parser = argparse.ArgumentParser(
            prog="swe_care.harness",
            description="Evaluation tools for SWE-CARE",
            parents=[global_parser],
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        for cmd, info in SUBCOMMAND_MAP.items():
            subparsers.add_parser(cmd, help=info["help"])

        return parser.parse_args(args)

    # Create the appropriate subcommand parser with global parser as parent
    match subcommand:
        case "code_review_eval":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.harness {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            sub_parser.add_argument(
                "--dataset-file",
                type=Path,
                required=True,
                help="Path to the dataset file (code_review_task_instances.jsonl)",
            )
            sub_parser.add_argument(
                "--predictions-path",
                type=Path,
                required=True,
                help="Path to the predictions file or directory containing predictions",
            )
            sub_parser.add_argument(
                "--evaluator",
                type=EvaluatorType,
                nargs="+",
                required=True,
                choices=[e.value for e in EvaluatorType],
                help="Evaluator type to use",
            )

            available_providers, available_models = get_available_models_and_providers()

            sub_parser.add_argument(
                "--model",
                type=str,
                required=False,
                choices=available_models,
                help=f"Model name to use for LLM evaluation. Available models: {', '.join(available_models)}",
            )
            sub_parser.add_argument(
                "--model-provider",
                type=str,
                required=False,
                choices=available_providers,
                help=f"Model provider for LLM evaluation. Available providers: {', '.join(available_providers)}",
            )
            sub_parser.add_argument(
                "--model-args",
                type=str,
                required=False,
                default=None,
                help="Comma-separated model arguments for LLM evaluation (e.g., 'temperature=0.7,top_p=0.9')",
            )
            sub_parser.add_argument(
                "--jobs",
                type=int,
                default=2,
                help="Number of parallel jobs to run (default: 2)",
            )

    # Parse all arguments with the subcommand parser
    # This will include both global and subcommand-specific arguments
    # Remove the subcommand itself from args
    args_without_subcommand = args[:subcommand_index] + args[subcommand_index + 1 :]
    final_namespace = sub_parser.parse_args(args_without_subcommand)
    final_namespace.command = subcommand

    return final_namespace


def main():
    args = get_args()

    if args.command in SUBCOMMAND_MAP:
        # Get the function from the mapping
        cmd_info = SUBCOMMAND_MAP[args.command]
        function = cmd_info["function"]

        # Prepare common arguments
        common_kwargs = {"output_dir": args.output_dir}

        # Add specific arguments based on subcommand
        match args.command:
            case "code_review_eval":
                function(
                    dataset_file=args.dataset_file,
                    predictions_path=args.predictions_path,
                    evaluator_types=args.evaluator,
                    model=args.model,
                    model_provider=args.model_provider,
                    model_args=args.model_args,
                    jobs=args.jobs,
                    **common_kwargs,
                )
    else:
        logger.info("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()

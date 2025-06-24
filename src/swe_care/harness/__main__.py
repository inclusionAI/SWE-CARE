import argparse
import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

from swe_care.harness.code_review_eval import EvaluatorType, code_review_eval

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "code_review_eval": {
        "function": code_review_eval,
        "help": "Run evaluation on code review predictions",
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
                description="Run evaluation on code review predictions",
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
            sub_parser.add_argument(
                "--llm-model",
                type=str,
                required=False,
                help="LLM model to use",
            )
            sub_parser.add_argument(
                "--llm-api-key",
                type=str,
                required=False,
                help="LLM API key to use",
            )
            sub_parser.add_argument(
                "--llm-base-url",
                type=str,
                required=False,
                help="LLM API base to use",
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
                llm_kwargs = {}
                if args.llm_api_key:
                    llm_kwargs["api_key"] = args.llm_api_key
                if args.llm_base_url:
                    llm_kwargs["base_url"] = args.llm_base_url
                llm_client = OpenAI(**llm_kwargs)
                logger.info(
                    f"Using LLM client: {llm_client} with model: {args.llm_model}"
                )

                function(
                    dataset_file=args.dataset_file,
                    predictions_path=args.predictions_path,
                    evaluator_types=args.evaluator,
                    llm_client=llm_client,
                    llm_model=args.llm_model,
                    **common_kwargs,
                )
    else:
        logger.info("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()

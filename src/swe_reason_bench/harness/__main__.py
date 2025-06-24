import argparse
import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

from swe_reason_bench.harness.code_review_eval import EvaluatorType, code_review_eval

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "code_review_eval": {
        "function": code_review_eval,
        "help": "Run evaluation on code review predictions",
    },
}


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

    if subcommand is None:
        # No subcommand found, use normal argparse
        parser = argparse.ArgumentParser(
            prog="swe_reason_bench.harness",
            description="Evaluation tools for SWE Reason Bench",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Path to output directory",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        for cmd, info in SUBCOMMAND_MAP.items():
            subparsers.add_parser(cmd, help=info["help"])

        return parser.parse_args(args)

    # Split arguments into global and subcommand parts
    global_args = args[:subcommand_index] + args[subcommand_index + 1 :]

    # Parse global arguments
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--output-dir", type=Path)

    global_namespace, remaining_args = global_parser.parse_known_args(global_args)

    # Create the appropriate subcommand parser
    match subcommand:
        case "code_review_eval":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_reason_bench.harness {subcommand}"
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

    # Parse subcommand arguments
    sub_namespace = sub_parser.parse_args(remaining_args)

    # Combine namespaces
    final_namespace = argparse.Namespace()
    final_namespace.command = subcommand

    # Add global arguments
    final_namespace.output_dir = global_namespace.output_dir

    # Add subcommand arguments
    for key, value in vars(sub_namespace).items():
        setattr(final_namespace, key, value)

    # Ensure output_dir is provided
    if not final_namespace.output_dir:
        logger.error("the following arguments are required: --output-dir")
        sys.exit(2)

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

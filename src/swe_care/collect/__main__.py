import argparse
import sys
from pathlib import Path

from loguru import logger

import swe_care.collect.build_code_review_dataset
import swe_care.collect.classify_prs_data
import swe_care.collect.get_graphql_prs_data
import swe_care.collect.get_top_repos
from swe_care.collect.build_code_review_dataset import build_code_review_dataset
from swe_care.collect.classify_prs_data import classify_prs_data
from swe_care.collect.get_graphql_prs_data import get_graphql_prs_data
from swe_care.collect.get_top_repos import get_top_repos

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "get_top_repos": {
        "function": get_top_repos,
        "help": swe_care.collect.get_top_repos.__doc__,
    },
    "get_graphql_prs_data": {
        "function": get_graphql_prs_data,
        "help": swe_care.collect.get_graphql_prs_data.__doc__,
    },
    "classify_prs_data": {
        "function": classify_prs_data,
        "help": swe_care.collect.classify_prs_data.__doc__,
    },
    "build_code_review_dataset": {
        "function": build_code_review_dataset,
        "help": swe_care.collect.build_code_review_dataset.__doc__,
    },
}


def create_global_parser():
    """Create a parser with global arguments that can be used as a parent parser."""
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--tokens",
        type=str,
        nargs="*",
        default=None,
        help="GitHub API token(s) to be used randomly for fetching data",
    )
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
            prog="swe_care.collect",
            description="Data collection tools for SWE-CARE",
            parents=[global_parser],
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        for cmd, info in SUBCOMMAND_MAP.items():
            subparsers.add_parser(cmd, help=info["help"])

        return parser.parse_args(args)

    # Create the appropriate subcommand parser with global parser as parent
    match subcommand:
        case "get_top_repos":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            sub_parser.add_argument(
                "--language",
                type=str,
                required=True,
                help="Programming language to search for",
            )
            sub_parser.add_argument(
                "--top-n",
                type=int,
                required=True,
                help="Number of top repositories to fetch",
            )

        case "get_graphql_prs_data":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            repo_group = sub_parser.add_mutually_exclusive_group(required=True)
            repo_group.add_argument(
                "--repo-file",
                type=Path,
                help="Path to repository JSONL file containing repository information, each line should be a JSON object with at least a 'name' field in format 'owner/repo'. Optionally, include 'pr_cursor' field to resume fetching from a specific cursor for each repository.",
                default=None,
            )
            repo_group.add_argument(
                "--repo",
                type=str,
                help="Repository in format 'owner/repo'",
                default=None,
            )
            sub_parser.add_argument(
                "--max-number",
                type=int,
                default=None,
                help="Maximum number of PRs to fetch per page (ignored when specific_prs is provided). If not provided, all PRs will be fetched.",
            )
            sub_parser.add_argument(
                "--jobs",
                type=int,
                default=2,
                help="Number of concurrent jobs/threads to use (default: 2)",
            )
            sub_parser.add_argument(
                "--specific-prs",
                type=int,
                nargs="*",
                default=None,
                help="Specific PR numbers to fetch (if not specified, fetches all PRs with closing issues)",
            )
            sub_parser.add_argument(
                "--after-pr-cursor",
                type=str,
                default=None,
                help="Resume fetching PRs after this cursor (useful for resuming interrupted runs). When used with --repo-file, acts as fallback for repositories without pr_cursor field in the file.",
            )
        case "classify_prs_data":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            sub_parser.add_argument(
                "--graphql-prs-data-file",
                type=Path,
                required=True,
                help="Path to GraphQL PRs data file or directory containing *_graphql_prs_data.jsonl files",
            )
            sub_parser.add_argument(
                "--jobs",
                type=int,
                default=2,
                help="Number of concurrent jobs/threads to use (default: 2)",
            )
        case "build_code_review_dataset":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description=SUBCOMMAND_MAP[subcommand]["help"],
            )
            sub_parser.add_argument(
                "--graphql-prs-data-file",
                type=Path,
                required=True,
                help="Path to GraphQL PRs data file or directory containing *_graphql_prs_data.jsonl files",
            )
            sub_parser.add_argument(
                "--pr-classification-file",
                type=Path,
                required=True,
                help="Path to PR classification file or directory containing *_pr_classification.jsonl files",
            )
            sub_parser.add_argument(
                "--skip-existing",
                action="store_true",
                default=False,
                help="Skip processing existing instance_id in the output file (default: False)",
            )
            sub_parser.add_argument(
                "--jobs",
                type=int,
                default=2,
                help="Number of concurrent jobs/threads to use (default: 2)",
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
        common_kwargs = {"output_dir": args.output_dir, "tokens": args.tokens}

        # Add specific arguments based on subcommand
        match args.command:
            case "get_top_repos":
                function(language=args.language, top_n=args.top_n, **common_kwargs)
            case "get_graphql_prs_data":
                function(
                    repo_file=args.repo_file,
                    repo=args.repo,
                    max_number=args.max_number,
                    specific_prs=args.specific_prs,
                    jobs=args.jobs,
                    after_pr_cursor=args.after_pr_cursor,
                    **common_kwargs,
                )
            case "classify_prs_data":
                function(
                    graphql_prs_data_file=args.graphql_prs_data_file,
                    jobs=args.jobs,
                    **common_kwargs,
                )
            case "build_code_review_dataset":
                function(
                    graphql_prs_data_file=args.graphql_prs_data_file,
                    pr_classification_file=args.pr_classification_file,
                    skip_existing=args.skip_existing,
                    jobs=args.jobs,
                    **common_kwargs,
                )
    else:
        logger.info("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()

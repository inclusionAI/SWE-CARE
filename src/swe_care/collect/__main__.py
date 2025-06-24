import argparse
import sys
from pathlib import Path

from loguru import logger

from swe_care.collect.build_code_review_dataset import (
    build_code_review_dataset,
)
from swe_care.collect.evaluate_commits import (
    evaluate_commits,
)
from swe_care.collect.get_graphql_prs_data import (
    get_graphql_prs_data,
)
from swe_care.collect.get_top_repos import get_top_repos

# Mapping of subcommands to their function names
SUBCOMMAND_MAP = {
    "get_top_repos": {
        "function": get_top_repos,
        "help": "Get top repositories for a given language",
    },
    "get_graphql_prs_data": {
        "function": get_graphql_prs_data,
        "help": "Get PR data from GitHub GraphQL API",
    },
    "evaluate_commits": {
        "function": evaluate_commits,
        "help": "Evaluate commits in PRs using heuristic rules",
    },
    "build_code_review_dataset": {
        "function": build_code_review_dataset,
        "help": "Build code review task dataset",
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
                description="Get top repositories for a given language",
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
                description="Get PR data from GitHub GraphQL API",
            )
            repo_group = sub_parser.add_mutually_exclusive_group(required=True)
            repo_group.add_argument(
                "--repo-file", type=Path, help="Path to repository file"
            )
            repo_group.add_argument(
                "--repo", type=str, help="Repository in format 'owner/repo'"
            )
            sub_parser.add_argument(
                "--max-number",
                type=int,
                default=10,
                help="Maximum number of PRs to fetch per page",
            )
        case "evaluate_commits":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description="Evaluate commits in PRs using heuristic rules",
            )
            sub_parser.add_argument(
                "--graphql-prs-data-file",
                type=Path,
                required=True,
                help="Path to GraphQL PRs data file",
            )
        case "build_code_review_dataset":
            sub_parser = argparse.ArgumentParser(
                prog=f"swe_care.collect {subcommand}",
                parents=[global_parser],
                description="Build code review task dataset",
            )
            sub_parser.add_argument(
                "--graphql-prs-data-file",
                type=Path,
                required=True,
                help="Path to GraphQL PRs data file",
            )
            sub_parser.add_argument(
                "--pr-commits-evaluation-file",
                type=Path,
                required=True,
                help="Path to PR commits evaluation file",
            )
            sub_parser.add_argument(
                "--skip-existing",
                action="store_true",
                default=False,
                help="Skip processing existing instance_id in the output file (default: False)",
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
                    repo_file=getattr(args, "repo_file", None),
                    repo=getattr(args, "repo", None),
                    max_number=getattr(args, "max_number", 10),
                    **common_kwargs,
                )
            case "evaluate_commits":
                logger.warning("tokens are not used for evaluate_commits")
                common_kwargs.pop("tokens")
                function(
                    graphql_prs_data_file=args.graphql_prs_data_file,
                    **common_kwargs,
                )
            case "build_code_review_dataset":
                function(
                    graphql_prs_data_file=args.graphql_prs_data_file,
                    pr_commits_evaluation_file=args.pr_commits_evaluation_file,
                    skip_existing=getattr(args, "skip_existing", False),
                    **common_kwargs,
                )
    else:
        logger.info("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()

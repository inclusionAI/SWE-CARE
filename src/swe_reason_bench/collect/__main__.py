import argparse
import sys
from pathlib import Path

from swe_reason_bench.collect.build_code_review_dataset import (
    build_code_review_dataset,
)
from swe_reason_bench.collect.get_graphql_prs_data import (
    get_graphql_prs_data,
)
from swe_reason_bench.collect.get_top_repos import get_top_repos

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
    "build_code_review_dataset": {
        "function": build_code_review_dataset,
        "help": "Build code review task dataset",
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
            prog="swe_reason_bench.collect",
            description="Data collection tools for SWE Reason Bench",
        )
        parser.add_argument(
            "--tokens",
            type=str,
            nargs="*",
            default=None,
            help="GitHub API token(s) to be used randomly for fetching data",
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
    global_parser.add_argument("--tokens", type=str, nargs="*", default=None)
    global_parser.add_argument("--output-dir", type=Path)

    global_namespace, remaining_args = global_parser.parse_known_args(global_args)

    # Create the appropriate subcommand parser
    if subcommand == "get_top_repos":
        sub_parser = argparse.ArgumentParser(
            prog=f"swe_reason_bench.collect {subcommand}"
        )
        sub_parser.add_argument("--language", type=str, required=True)
        sub_parser.add_argument("--top_n", type=int, required=True)

    elif subcommand in ["build_code_review_dataset", "get_graphql_prs_data"]:
        sub_parser = argparse.ArgumentParser(
            prog=f"swe_reason_bench.collect {subcommand}"
        )
        repo_group = sub_parser.add_mutually_exclusive_group(required=True)
        repo_group.add_argument(
            "--repo-file", type=Path, help="Path to repository file"
        )
        repo_group.add_argument(
            "--repo", type=str, help="Repository in format 'owner/repo'"
        )
        if subcommand == "get_graphql_prs_data":
            sub_parser.add_argument(
                "--max-number",
                type=int,
                default=10,
                help="Maximum number of PRs to fetch per page",
            )

    # Parse subcommand arguments
    sub_namespace = sub_parser.parse_args(remaining_args)

    # Combine namespaces
    final_namespace = argparse.Namespace()
    final_namespace.command = subcommand

    # Add global arguments
    final_namespace.tokens = global_namespace.tokens
    final_namespace.output_dir = global_namespace.output_dir

    # Add subcommand arguments
    for key, value in vars(sub_namespace).items():
        setattr(final_namespace, key, value)

    # Ensure output_dir is provided
    if not final_namespace.output_dir:
        print("error: the following arguments are required: --output-dir")
        sys.exit(2)

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
        if args.command == "get_top_repos":
            function(language=args.language, top_n=args.top_n, **common_kwargs)
        elif args.command in [
            "build_code_review_dataset",
            "get_graphql_prs_data",
        ]:
            kwargs = {
                "repo_file": getattr(args, "repo_file", None),
                "repo": getattr(args, "repo", None),
                **common_kwargs,
            }
            if args.command == "get_graphql_prs_data":
                kwargs["max_number"] = getattr(args, "max_number", 10)
            function(**kwargs)
    else:
        print("Please specify a command. Use --help for available commands.")


if __name__ == "__main__":
    main()

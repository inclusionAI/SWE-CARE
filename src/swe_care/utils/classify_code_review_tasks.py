#!/usr/bin/env python3
"""
Script to classify code review task dataset for problem domains and estimated review effort.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

from loguru import logger

from swe_care.utils.llm_models import init_llm_client
from swe_care.utils.load import load_code_review_dataset

# Thread-safe counter for progress tracking
progress_lock = Lock()
processed_count = 0


def classify_problem_domain(client, problem_statement: str) -> str:
    """Classify the problem domain of a code review task."""

    system_prompt = """You are an expert software engineer responsible for classifying software development tasks. Your task is to analyze a problem statement and classify it into exactly one of the following categories:

1. Bug Fixes: Resolving functional errors, crashes, incorrect outputs
2. New Feature Additions: Adding new functionality or features to the application
3. Code Refactoring / Architectural Improvement: Improving code structure, readability, maintainability without changing external behavior
4. Documentation Updates: Changes related to code comments or external documentation
5. Test Suite / CI Enhancements: Improving test coverage, test quality, or continuous integration processes
6. Performance Optimizations: Improving application speed, response time, or resource usage efficiency
7. Security Patches / Vulnerability Fixes: Fixing code defects that could lead to security issues
8. Dependency Updates & Env Compatibility: Updating third-party library dependencies or ensuring compatibility across different environments
9. Code Style, Linting, Formatting Fixes: Ensuring code complies with team coding standards and consistency

Please respond with ONLY the category name exactly as listed above (e.g., "Bug Fixes", "New Feature Additions", etc.)."""

    user_prompt = f"""Please classify the following problem statement into one of the predefined categories:

Problem Statement:
{problem_statement}

Category:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages)
        return response.strip()
    except Exception as e:
        logger.error(f"Error classifying problem domain: {e}")
        return "Unknown"


def estimate_review_effort(
    client, title: str, body: str, commit_message: str, patch: str
) -> int:
    """Estimate the review effort for a code review task (1-5 scale)."""

    system_prompt = """You are an experienced software engineer responsible for estimating code review effort. Your task is to estimate how much effort would be required to review a code change on a scale of 1 to 5, where:

1 = Very Low Effort: Simple changes like typo fixes, minor documentation updates, or trivial formatting changes
2 = Low Effort: Small bug fixes, minor feature additions, or straightforward code changes affecting a few lines
3 = Medium Effort: Moderate complexity changes involving multiple files, standard feature implementations, or routine refactoring
4 = High Effort: Complex changes affecting multiple components, significant new features, or architectural modifications requiring careful review
5 = Very High Effort: Major architectural changes, complex algorithms, security-critical modifications, or changes requiring domain expertise

Consider factors like:
- Size and scope of the change
- Complexity of the code modifications
- Number of files affected
- Potential impact on system behavior
- Risk level of the changes

Please respond with ONLY a single number from 1 to 5."""

    user_prompt = f"""Please estimate the review effort (1-5) for the following code change:

**Pull Request Title:**
{title}

**Pull Request Description:**
{body}

**Commit Message:**
{commit_message}

**Code Changes (Patch):**
{patch}

Review Effort (1-5):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages)
        effort = int(response.strip())
        # Ensure the effort is within valid range
        return max(1, min(5, effort))
    except (ValueError, Exception) as e:
        logger.error(f"Error estimating review effort: {e}")
        return 3  # Default to medium effort


def estimate_review_difficulty(
    client, title: str, body: str, commit_message: str, changed_files: int, patch: str
) -> int:
    """Estimate the review difficulty for a code review task (1-3 scale)."""

    system_prompt = """You are an experienced software engineer responsible for estimating code review difficulty. Your task is to estimate how difficult it would be to review a code change on a scale of 1 to 3, where:

1 = Low Difficulty: Small changes with minimal complexity, good test coverage, low impact, and easy to understand.
2 = Medium Difficulty: Small bug fixes, minor feature additions, or straightforward code changes affecting a few lines
3 = High Difficulty: Moderate size changes, some complexity in the code, some gaps in test coverage, moderate impact, and requires more time for review.

Consider factors like:
- Pull request size and scope
- Complexity of the code modifications
- Number of files affected
- Potential impact on system behavior and codebase
- Test coverage and quality of tests
- Clarity and quality of the code changes

Please respond with ONLY a single number from 1 to 5."""

    user_prompt = f"""Please estimate the difficulty level (Low, Medium, High) for the following code change:

**Pull Request Title:**
{title}

**Pull Request Description:**
{body}

**Commit Message:**
{commit_message}

**Changed Files:**
{str(changed_files)}

**Code Changes (Patch):**
{patch}

Difficulty (Low, Medium, High):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages)
        difficulty = int(response.strip())
        # Ensure the difficulty is within valid range
        return max(1, min(3, difficulty))
    except (ValueError, Exception) as e:
        logger.error(f"Error estimating review difficulty: {e}")
        return 2  # Default to medium difficulty


def process_single_instance(args: Tuple[Any, int, int]) -> Dict[str, Any]:
    """Process a single code review instance.

    Args:
        args: Tuple containing (instance, index, total_count)

    Returns:
        Dictionary with classification results
    """
    global processed_count
    instance, index, total_count = args

    try:
        # Create a new client for this thread to avoid conflicts
        client = init_llm_client(
            model="gemini-2.5-pro-preview-06-05",
            model_provider="openai",
            temperature=0.1,
        )

        # Classify problem domain
        problem_domain = classify_problem_domain(client, instance.problem_statement)

        # Estimate review effort
        review_effort = estimate_review_effort(
            client,
            instance.title,
            instance.body,
            instance.commit_to_review.head_commit_message,
            instance.commit_to_review.patch_to_review,
        )

        # Thread-safe progress update
        with progress_lock:
            processed_count += 1
            logger.info(
                f"Completed instance {processed_count}/{total_count}: {instance.instance_id} "
                f"(Domain: {problem_domain}, Effort: {review_effort})"
            )

        # Store result
        result = {
            "instance_id": instance.instance_id,
            "problem_domains": problem_domain,
            "estimated_review_effort": review_effort,
        }
        return result

    except Exception as e:
        with progress_lock:
            processed_count += 1
            logger.error(f"Error processing instance {instance.instance_id}: {e}")

        # Return default result on error
        result = {
            "instance_id": instance.instance_id,
            "problem_domains": "Unknown",
            "estimated_review_effort": 3,
        }
        return result


def main(max_workers: int = None, dry_run: bool = False):
    """Main function to classify code review tasks.

    Args:
        max_workers: Maximum number of worker threads (default: min(10, num_instances))
        dry_run: If True, process only first 3 instances to test setup
    """
    global processed_count
    processed_count = 0

    # Setup logging
    if dry_run:
        logger.info("Starting DRY RUN - processing only first 3 instances...")
    else:
        logger.info("Starting code review task classification with multi-threading...")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        return

    # Load dataset
    dataset_file = Path(
        "results/manual_selected_swe_bench_verified/dataset/code_review_task_instances.jsonl"
    )
    try:
        instances = load_code_review_dataset(dataset_file)

        if dry_run:
            instances = instances[:3]  # Limit to first 3 instances for testing
            logger.info(f"DRY RUN: Processing only {len(instances)} instances")
        else:
            logger.info(f"Loaded {len(instances)} instances")

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Test client initialization
    try:
        test_client = init_llm_client(
            model="gemini-2.5-pro-preview-06-05",
            model_provider="openai",
            temperature=0.1,
        )
        print(test_client)
        logger.info("Successfully initialized OpenAI client")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return

    # Configure thread pool size
    if max_workers is None:
        max_workers = min(10, len(instances))  # Default limit to 10 concurrent threads
    else:
        max_workers = min(max_workers, len(instances))  # Don't exceed instance count

    if dry_run:
        max_workers = min(2, max_workers)  # Use fewer threads for dry run

    logger.info(f"Using {max_workers} worker threads for processing")

    # Prepare arguments for thread pool
    task_args = [(instance, i, len(instances)) for i, instance in enumerate(instances)]

    # Process instances using thread pool
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(process_single_instance, args): args[0]
            for args in task_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                instance = future_to_instance[future]
                logger.error(f"Future failed for instance {instance.instance_id}: {e}")
                # Add default result for failed future
                default_result = {
                    "instance_id": instance.instance_id,
                    "problem_domains": "Unknown",
                    "estimated_review_effort": 3,
                }
                results.append(default_result)

    # Save results to JSONL file
    output_file = Path(
        "code_review_classification_results_dry_run.jsonl"
        if dry_run
        else "code_review_classification_results.jsonl"
    )
    try:
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        logger.success(f"Classification results saved to {output_file}")
        logger.info(f"Processed {len(results)} instances")

        # Print summary statistics
        domain_counts = {}
        effort_counts = {}

        for result in results:
            domain = result["problem_domains"]
            effort = result["estimated_review_effort"]

            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            effort_counts[effort] = effort_counts.get(effort, 0) + 1

        logger.info("Problem Domain Distribution:")
        for domain, count in sorted(domain_counts.items()):
            logger.info(f"  {domain}: {count}")

        logger.info("Review Effort Distribution:")
        for effort, count in sorted(effort_counts.items()):
            logger.info(f"  {effort}: {count}")

        if dry_run:
            logger.info(
                "DRY RUN completed successfully! You can now run the full classification."
            )

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Classify code review tasks using multi-threading"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: min(10, num_instances))",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only first 3 instances to test setup (saves to *_dry_run.jsonl)",
    )
    args = parser.parse_args()

    # Check if dataset file exists
    dataset_file = Path(
        "results/manual_selected_swe_bench_verified/dataset/code_review_task_instances.jsonl"
    )
    if not dataset_file.exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        print(
            "Please make sure the dataset file exists before running the classification."
        )
        sys.exit(1)

    if args.dry_run:
        print("Starting code review task classification DRY RUN...")
    else:
        print("Starting code review task classification...")
    main(max_workers=args.workers, dry_run=args.dry_run)

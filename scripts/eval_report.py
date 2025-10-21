#!/usr/bin/env python3
"""
Generate comprehensive evaluation report from SWE-CARE pipeline results.

This script analyzes evaluation results from multiple models and settings,
calculating performance metrics across different dimensions.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from swe_care.schema.dataset import CodeReviewTaskInstance
from swe_care.schema.evaluation import CodeReviewEvaluationResult
from swe_care.utils.load import load_code_review_dataset, load_code_review_eval_result


def parse_eval_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse evaluation result filename to extract metadata.

    Expected patterns:
    - {dataset_name}__{file_source}__{model_name}_report_{timestamp}.jsonl
    - {dataset_name}__bm25__k{k}__{model_name}_report_{timestamp}.jsonl
    - {dataset_name}__{file_source}__skeleton__{model_name}_report_{timestamp}.jsonl
    - {dataset_name}__bm25__k{k}__skeleton__{model_name}_report_{timestamp}.jsonl

    Args:
        filename: The filename to parse

    Returns:
        Dictionary with parsed metadata or None if parsing fails
    """
    # Remove .jsonl extension
    name_parts = filename.replace(".jsonl", "")

    # Support optional __skeleton suffix before model name
    # Pattern for files with bm25 and k value, optional __skeleton
    bm25_pattern = r"^(.+?)__bm25__k(\d+)(__skeleton)?__(.+?)_report_(\d{8}_\d{6})$"
    # Pattern for files without k value (oracle, none, all), optional __skeleton
    basic_pattern = (
        r"^(.+?)__(oracle|none|all)(__skeleton)?__(.+?)_report_(\d{8}_\d{6})$"
    )

    bm25_match = re.match(bm25_pattern, name_parts)
    if bm25_match:
        return {
            "dataset_name": bm25_match.group(1),
            "file_source": "bm25",
            "k": int(bm25_match.group(2)),
            "skeleton": bm25_match.group(3) is not None,
            "model_name": bm25_match.group(4),
            "timestamp": bm25_match.group(5),
            "datetime": datetime.strptime(bm25_match.group(5), "%Y%m%d_%H%M%S"),
        }

    basic_match = re.match(basic_pattern, name_parts)
    if basic_match:
        return {
            "dataset_name": basic_match.group(1),
            "file_source": basic_match.group(2),
            "k": None,
            "skeleton": basic_match.group(3) is not None,
            "model_name": basic_match.group(4),
            "timestamp": basic_match.group(5),
            "datetime": datetime.strptime(basic_match.group(5), "%Y%m%d_%H%M%S"),
        }

    logger.warning(f"Could not parse filename: {filename}")
    return None


def get_file_source_key(file_source: str, k: Optional[int], skeleton: bool) -> str:
    """Generate a consistent key for file source settings."""
    base = f"bm25_k{k}" if file_source == "bm25" and k is not None else file_source
    return f"{base}_skeleton" if skeleton else base


def collect_eval_results(
    eval_output_dir: Path, dataset_instances: List[CodeReviewTaskInstance]
) -> Dict[str, Dict[str, Dict[str, CodeReviewEvaluationResult]]]:
    """Collect all evaluation results from the output directory.

    Returns nested dict: model -> file_source_setting -> instance_id -> result
    """
    results = defaultdict(lambda: defaultdict(dict))
    instance_ids = {inst.instance_id for inst in dataset_instances}

    # Scan all subdirectories (one per model)
    for model_dir in eval_output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        logger.info(f"Processing results for model: {model_name}")

        # Group files by file source setting
        files_by_setting = defaultdict(list)

        for result_file in model_dir.glob("*.jsonl"):
            parsed = parse_eval_filename(result_file.name)
            if parsed:
                file_source_key = get_file_source_key(
                    parsed["file_source"], parsed["k"], parsed.get("skeleton", False)
                )
                files_by_setting[file_source_key].append(
                    {"path": result_file, "parsed": parsed}
                )

        # For each setting, load results and keep only the latest for each instance
        for setting, files in files_by_setting.items():
            # Sort files by timestamp (latest first)
            files.sort(key=lambda x: x["parsed"]["datetime"], reverse=True)

            seen_instances = set()

            for file_info in files:
                try:
                    eval_results = load_code_review_eval_result(file_info["path"])

                    for result in eval_results:
                        # Only keep if instance is in dataset and not seen yet
                        if (
                            result.instance_id in instance_ids
                            and result.instance_id not in seen_instances
                        ):
                            results[model_name][setting][result.instance_id] = result
                            seen_instances.add(result.instance_id)

                except Exception as e:
                    logger.error(f"Error loading {file_info['path']}: {e}")

    return results


def calculate_average_score(scores: List[float]) -> float:
    """Calculate average score, handling empty lists."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def calculate_evaluator_scores(
    results: List[CodeReviewEvaluationResult],
) -> Dict[str, float]:
    """Calculate average scores by evaluator type."""
    evaluator_scores = defaultdict(list)

    for result in results:
        for evaluation in result.evaluations:
            evaluator_name = evaluation.evaluator
            if evaluation.evaluation.get("score") is not None:
                evaluator_scores[evaluator_name].append(evaluation.evaluation["score"])

    return {
        evaluator: calculate_average_score(scores)
        for evaluator, scores in evaluator_scores.items()
    }


def calculate_metadata_scores(
    results: List[Tuple[CodeReviewTaskInstance, Optional[CodeReviewEvaluationResult]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Calculate scores grouped by metadata categories."""
    metadata_groups = {
        "problem_domain": defaultdict(list),
        "difficulty": defaultdict(list),
        "estimated_review_effort": defaultdict(list),
    }

    for instance, result in results:
        score = result.score if result else 0.0

        # Problem domain
        if instance.metadata.problem_domain:
            metadata_groups["problem_domain"][instance.metadata.problem_domain].append(
                score
            )

        # Difficulty
        if instance.metadata.difficulty:
            metadata_groups["difficulty"][instance.metadata.difficulty].append(score)

        # Estimated review effort
        if instance.metadata.estimated_review_effort is not None:
            effort_key = f"effort_{instance.metadata.estimated_review_effort}"
            metadata_groups["estimated_review_effort"][effort_key].append(score)

    # Calculate statistics for each group
    metadata_stats = {}
    for category, groups in metadata_groups.items():
        metadata_stats[category] = {}
        for group_name, scores in groups.items():
            metadata_stats[category][group_name] = {
                "average_score": calculate_average_score(scores),
                "count": len(scores),
            }

    return metadata_stats


def generate_report(
    dataset_name_or_path: Path | str, eval_output_dir: Path, report_output_file: Path
) -> None:
    """Generate comprehensive evaluation report."""
    logger.info(f"Loading dataset from {dataset_name_or_path}")
    dataset_instances = load_code_review_dataset(dataset_name_or_path)
    total_instances = len(dataset_instances)
    logger.info(f"Loaded {total_instances} dataset instances")

    logger.info(f"Collecting evaluation results from {eval_output_dir}")
    all_results = collect_eval_results(eval_output_dir, dataset_instances)

    # Build report structure
    report = {
        "metadata": {
            "dataset_name_or_path": str(dataset_name_or_path),
            "eval_output_dir": str(eval_output_dir),
            "total_instances": total_instances,
            "generation_timestamp": datetime.now().isoformat(),
        },
        "model_results": {},
        "rankings": {},
    }

    # Process results for each model
    model_setting_scores = []  # For overall ranking

    for model_name, model_results in all_results.items():
        logger.info(f"Processing model: {model_name}")

        model_report = {"settings": {}}

        for setting, setting_results in model_results.items():
            logger.info(f"  Processing setting: {setting}")

            # Match instances with results
            matched_results = []
            missing_instances = []

            for instance in dataset_instances:
                result = setting_results.get(instance.instance_id)
                matched_results.append((instance, result))
                if result is None:
                    missing_instances.append(instance.instance_id)

            # Calculate overall average score
            scores = [r.score for _, r in matched_results if r is not None]
            scores.extend([0.0] * len(missing_instances))  # Add 0 for missing
            average_score = calculate_average_score(scores)

            # Calculate scores by evaluator
            existing_results = [r for _, r in matched_results if r is not None]
            evaluator_scores = calculate_evaluator_scores(existing_results)

            # Calculate scores by metadata
            metadata_scores = calculate_metadata_scores(matched_results)

            model_report["settings"][setting] = {
                "average_score": average_score,
                "evaluated_instances": len(existing_results),
                "missing_instances": len(missing_instances),
                "missing_instance_ids": missing_instances[:10],  # Show first 10
                "evaluator_scores": evaluator_scores,
                "metadata_scores": metadata_scores,
            }

            # Add to ranking list
            model_setting_scores.append(
                {
                    "model": model_name,
                    "setting": setting,
                    "average_score": average_score,
                    "evaluated_ratio": len(existing_results) / total_instances,
                }
            )

        report["model_results"][model_name] = model_report

    # Sort rankings by average score
    model_setting_scores.sort(key=lambda x: x["average_score"], reverse=True)
    report["rankings"]["by_average_score"] = model_setting_scores

    # Calculate best settings for each model
    best_by_model = {}
    for item in model_setting_scores:
        model = item["model"]
        if (
            model not in best_by_model
            or item["average_score"] > best_by_model[model]["average_score"]
        ):
            best_by_model[model] = item

    report["rankings"]["best_setting_per_model"] = list(best_by_model.values())

    # Save report
    logger.info(f"Saving report to {report_output_file}")
    report_output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    logger.success("Report generation complete!")
    logger.info(f"Processed {len(all_results)} models")
    logger.info(f"Total dataset instances: {total_instances}")

    print("\n=== Top 10 Model-Setting Combinations ===")
    for i, item in enumerate(model_setting_scores[:10], 1):
        print(
            f"{i}. {item['model']} ({item['setting']}): "
            f"Score={item['average_score']:.4f}, "
            f"Coverage={item['evaluated_ratio']:.1%}"
        )

    # Print skeleton comparison summary if available
    if report.get("skeleton_analysis", {}).get("by_model"):
        comps = sorted(
            report["skeleton_analysis"]["by_model"],
            key=lambda x: x["delta"],
            reverse=True,
        )
        print("\n=== Skeleton vs Non-Skeleton (by model/setting base) ===")
        for i, e in enumerate(comps[:10], 1):
            print(
                f"{i}. {e['model']} [{e['setting_base']}]: "
                f"with={e['with_skeleton']:.4f}, without={e['without_skeleton']:.4f}, "
                f"delta={e['delta']:+.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive evaluation report from SWE-CARE pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/eval_report.py \\
    --dataset-name-or-path results/dataset/code_review_task_instances.jsonl \\
    --eval-output-dir results/pipeline_output/evaluation \\
    --report-output-file results/evaluation_report.json

The script will:
1. Load all evaluation results from the output directory
2. Match results with dataset instances
3. Calculate average scores by model, setting, and metadata categories
4. Generate a comprehensive JSON report with rankings
5. Assign score 0 to missing instances for fair comparison
""",
    )

    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        required=False,
        default="inclusionAI/SWE-CARE",
        help="Path to the dataset file or Hugging Face dataset name (default: inclusionAI/SWE-CARE)",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation results (organized by model)",
    )
    parser.add_argument(
        "--report-output-file",
        type=Path,
        required=True,
        help="Path for the output JSON report file",
    )

    args = parser.parse_args()

    if not args.eval_output_dir.exists():
        logger.error(f"Evaluation output directory not found: {args.eval_output_dir}")
        return

    # Generate report
    generate_report(
        dataset_name_or_path=args.dataset_name_or_path,
        eval_output_dir=args.eval_output_dir,
        report_output_file=args.report_output_file,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Augment SWE-CARE evaluation results with optional reward-model scores, then:
1) Write evaluator-model-averaged results into `<eval output dir>/all`
2) Generate a unified JSON report for those grouped results.

If `--reward-model-scores-file` is provided:
- Writes an updated evaluation directory as a sibling of `--eval-output-dir`:
  `<eval output dir>/../evaluation_with_reward_model`
- Subsequent steps (grouping + report) run on that updated directory.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

import eval_report
from swe_care.schema.evaluation import CodeReviewEvaluationResult, EvaluatorResult
from swe_care.utils.load import load_code_review_eval_result


_REPORT_SUFFIX_RE = re.compile(r"^(.*)_report(?:_\d{8}_\d{6})?$")


def _predictions_stem_from_path(path: str) -> str:
    stem = Path(path).stem
    match = _REPORT_SUFFIX_RE.match(stem)
    return match.group(1) if match else stem


def _load_reward_model_scores(
    reward_model_scores_file: Path,
) -> dict[str, dict[str, float]]:
    scores_by_predictions_stem: dict[str, dict[str, float]] = defaultdict(dict)
    for line_num, line in enumerate(
        reward_model_scores_file.read_text().splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Skipping invalid JSON in reward-model scores at line {line_num}: {e}"
            )
            continue

        instance_id = record.get("instance_id")
        score = record.get("score")
        data_path = record.get("data_path")

        if instance_id is None or score is None or data_path is None:
            logger.warning(
                "Skipping reward-model score missing required fields: "
                f"line={line_num}, keys={sorted(record.keys())}"
            )
            continue

        if score == -1:
            continue

        predictions_stem = _predictions_stem_from_path(str(data_path))
        scores_by_predictions_stem[predictions_stem][str(instance_id)] = float(score)

    return dict(scores_by_predictions_stem)


def _get_evaluator_bucket(eval_output_dir: Path, report_file: Path) -> str | None:
    rel_parts = report_file.relative_to(eval_output_dir).parts
    if len(rel_parts) >= 3:
        return rel_parts[0]
    return None


def _iter_report_files(eval_output_dir: Path) -> list[Path]:
    report_files: list[Path] = []
    for report_file in eval_output_dir.rglob("*.jsonl"):
        if not report_file.is_file():
            continue
        rel_parts = report_file.relative_to(eval_output_dir).parts
        if "all" in rel_parts:
            continue
        if eval_report.parse_eval_filename(report_file.name) is None:
            continue
        report_files.append(report_file)
    return report_files


def _select_latest_report_files(eval_output_dir: Path) -> list[Path]:
    """Select at most 1 report per (evaluator-bucket, predictions-stem).

    This prevents double-counting when legacy timestamped reports exist.
    """
    best: dict[tuple[str | None, str], tuple[datetime, float, Path]] = {}
    for report_file in _iter_report_files(eval_output_dir):
        parsed = eval_report.parse_eval_filename(report_file.name)
        if parsed is None:
            continue
        evaluator_bucket = _get_evaluator_bucket(eval_output_dir, report_file)
        predictions_stem = _predictions_stem_from_path(report_file.name)
        mtime = report_file.stat().st_mtime
        key = (evaluator_bucket, predictions_stem)

        current = best.get(key)
        candidate = (parsed["datetime"], mtime, report_file)
        if current is None or candidate[:2] > current[:2]:
            best[key] = candidate

    return sorted((v[2] for v in best.values()), key=lambda p: str(p))


def _write_eval_results(
    results: list[CodeReviewEvaluationResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")


def _rectify_overall_score(evaluations: list[EvaluatorResult]) -> float:
    scores: list[float] = []
    for evaluation in evaluations:
        value = evaluation.evaluation.get("score")
        if value is None:
            continue
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return sum(scores) / len(scores) if scores else 0.0


def _write_evaluation_with_reward_model(
    eval_output_dir: Path, reward_model_scores_file: Path
) -> Path:
    output_dir = eval_output_dir.parent / "evaluation_with_reward_model"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_by_predictions_stem = _load_reward_model_scores(reward_model_scores_file)
    logger.info(
        f"Loaded reward-model scores for {len(scores_by_predictions_stem)} settings"
    )

    updated_files = 0
    updated_instances = 0
    skipped_existing_rm = 0

    for report_file in _select_latest_report_files(eval_output_dir):
        rel_path = report_file.relative_to(eval_output_dir)
        out_path = output_dir / rel_path

        predictions_stem = _predictions_stem_from_path(report_file.name)
        rm_scores = scores_by_predictions_stem.get(predictions_stem)

        if not rm_scores:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_file, out_path)
            continue

        results = load_code_review_eval_result(report_file)
        results_by_id = {r.instance_id: r for r in results}

        for instance_id, rm_score in rm_scores.items():
            result = results_by_id.get(instance_id)
            if result is None:
                continue

            if any(e.evaluator == "reward_model" for e in result.evaluations):
                skipped_existing_rm += 1
                continue

            result.evaluations.append(
                EvaluatorResult(
                    evaluator="reward_model",
                    evaluation={"score": rm_score},
                )
            )
            result.score = _rectify_overall_score(result.evaluations)
            updated_instances += 1

        _write_eval_results(results, out_path)
        updated_files += 1

    logger.success(
        "Wrote updated evaluation results with reward-model scores: "
        f"{output_dir} (files_updated={updated_files}, "
        f"instances_updated={updated_instances}, skipped_existing_rm={skipped_existing_rm})"
    )
    return output_dir


def _calculate_average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_grouped_results(eval_output_dir: Path) -> Path:
    grouped_dir = eval_output_dir / "all"
    if grouped_dir.exists():
        shutil.rmtree(grouped_dir)
    grouped_dir.mkdir(parents=True, exist_ok=True)

    # Group latest reports by predictions-stem; each list item corresponds to one
    # evaluator model (top-level folder under `eval_output_dir` in the new layout).
    reports_by_predictions_stem: dict[str, list[Path]] = defaultdict(list)
    parsed_by_predictions_stem: dict[str, dict[str, Any]] = {}

    for report_file in _select_latest_report_files(eval_output_dir):
        parsed = eval_report.parse_eval_filename(report_file.name)
        if parsed is None:
            continue
        predictions_stem = _predictions_stem_from_path(report_file.name)
        reports_by_predictions_stem[predictions_stem].append(report_file)
        parsed_by_predictions_stem.setdefault(predictions_stem, parsed)

    logger.info(
        f"Found {len(reports_by_predictions_stem)} settings to group across evaluators"
    )

    for predictions_stem, report_files in reports_by_predictions_stem.items():
        parsed = parsed_by_predictions_stem[predictions_stem]
        model_name = parsed["model_name"]

        instance_scores: dict[str, list[float]] = defaultdict(list)
        evaluator_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for report_file in report_files:
            for result in load_code_review_eval_result(report_file):
                instance_scores[result.instance_id].append(float(result.score))

                for evaluation in result.evaluations:
                    value = evaluation.evaluation.get("score")
                    if value is None:
                        continue
                    if isinstance(value, (int, float)):
                        evaluator_scores[result.instance_id][
                            evaluation.evaluator
                        ].append(float(value))

        grouped_results: list[CodeReviewEvaluationResult] = []
        for instance_id, scores in instance_scores.items():
            evals: list[EvaluatorResult] = []
            for evaluator_name, scores_by_eval in evaluator_scores[instance_id].items():
                evals.append(
                    EvaluatorResult(
                        evaluator=evaluator_name,
                        evaluation={"score": _calculate_average(scores_by_eval)},
                    )
                )
            evals.sort(key=lambda e: e.evaluator)

            grouped_results.append(
                CodeReviewEvaluationResult(
                    instance_id=instance_id,
                    score=_calculate_average(scores),
                    evaluations=evals,
                )
            )

        grouped_results.sort(key=lambda r: r.instance_id)

        output_path = grouped_dir / model_name / f"{predictions_stem}_report.jsonl"
        _write_eval_results(grouped_results, output_path)

    logger.success(f"Wrote grouped results to {grouped_dir}")
    return grouped_dir


def _write_report_all(eval_output_dir: Path, grouped_dir: Path) -> Path:
    report_output_file = eval_output_dir.parent / "report_all.json"
    eval_report.generate_report(
        dataset_name_or_path="inclusionAI/SWE-CARE",
        eval_output_dir=grouped_dir,
        report_output_file=report_output_file,
    )
    logger.success(f"Wrote report: {report_output_file}")
    return report_output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment SWE-CARE evaluation results and generate grouped report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Group existing evaluator results and write report_all.json
  python scripts/eval_benchmark.py --eval-output-dir results/exp/evaluation

  # First merge reward-model scores, then group and report
  python scripts/eval_benchmark.py \\
    --eval-output-dir results/exp/evaluation \\
    --reward-model-scores-file results/exp/reward_model_scores/scores.jsonl
""",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        required=True,
        help="Evaluation output directory (pipeline `.../evaluation`)",
    )
    parser.add_argument(
        "--reward-model-scores-file",
        type=Path,
        required=False,
        default=None,
        help="Optional JSONL with reward-model scores (skips RM merge if omitted)",
    )
    args = parser.parse_args()

    eval_output_dir: Path = args.eval_output_dir
    if not eval_output_dir.exists():
        raise FileNotFoundError(f"--eval-output-dir not found: {eval_output_dir}")

    if args.reward_model_scores_file is not None:
        reward_model_scores_file: Path = args.reward_model_scores_file
        if not reward_model_scores_file.exists():
            raise FileNotFoundError(
                f"--reward-model-scores-file not found: {reward_model_scores_file}"
            )
        eval_output_dir = _write_evaluation_with_reward_model(
            eval_output_dir=eval_output_dir,
            reward_model_scores_file=reward_model_scores_file,
        )

    grouped_dir = _write_grouped_results(eval_output_dir)
    _write_report_all(eval_output_dir, grouped_dir)


if __name__ == "__main__":
    main()

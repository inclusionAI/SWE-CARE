# Repository Guidelines

## Project Structure & Module Organization
- `src/swe_care/`: source package
  - `collect/` (GitHub data + dataset build CLIs)
  - `inference/` (text generation + API inference)
  - `harness/` (evaluation runners and evaluators)
  - `schema/`, `utils/`, `templates/` (data models, helpers, prompts)
- `scripts/`: orchestration utilities (e.g., `run_eval_pipeline.py`)
- `docs/`: documentation and demos
- `results/`: local outputs created by commands (not tracked)

## Build, Test, and Development Commands
- Install deps: `pip install uv && uv sync` (or `pip install -e .`)
- Pre-commit hooks: `pre-commit install` • Run all: `pre-commit run --all-files`
- Lint/format: `ruff check .` • `ruff format .`
- Quick pipeline:
  `python scripts/run_eval_pipeline.py --dataset-file results/dataset/code_review_task_instances.jsonl --output-dir results/pipeline_output --model gpt-4o --model-provider openai --file-source oracle`
- Module CLIs: `python -m swe_care.collect ...` | `python -m swe_care.inference ...` | `python -m swe_care.harness ...`

## Coding Style & Naming Conventions
- Python ≥ 3.10 with type hints; prefer dataclasses for schemas.
- Ruff (Black-like): 4-space indent, line length 88, double quotes (`pyproject.toml`).
- snake_case for modules/functions/options; PascalCase for classes.
- Keep functions cohesive; shared helpers in `utils/`; prompts in `templates/`.
- Output naming examples: `<owner>__<repo>_graphql_prs_data.jsonl`, `<owner>__<repo>_rm_samples.jsonl`.

## Testing Guidelines
- No traditional unit tests; validate via small, reproducible runs that write to `results/`.
- Verify interfaces with help: `python -m swe_care.collect <sub> -h`.
- Aim for determinism (fixed params, stable sorting). Include sample outputs in PRs when logic changes.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`; add scope (e.g., `inference:`).
- PRs include: purpose, summary, example commands, sample outputs (paths under `results/`), and linked issues (e.g., `Closes #123`).
- Keep PRs focused; document new flags/env vars in README and `--help`. Ensure pre-commit passes.

## Security & Configuration Tips
- Never commit secrets. Use env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`; optional `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`.
- GitHub access via `--tokens`; watch rate limits.
- Retrieval with Pyserini may require Java 21; set `--retrieval-output-dir` for temporary work.

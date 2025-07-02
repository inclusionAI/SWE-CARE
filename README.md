# SWE-CARE: Software Engineering - Code Analysis and Review Evaluation

A comprehensive benchmark for evaluating Large Language Models (LLMs) on software engineering tasks, with a focus on code analysis, review, and issue-resolving capabilities.

## 📝 Overview

The primary goal of SWE-CARE is to assess LLMs in the following areas:

* **Solving Complex Programming Problems**: Evaluating the model's capability to understand, locate, and fix issues in real codebases.
* **Code Change Analysis**: Assessing the model's ability to analyze code changes, identify potential problems, and suggest improvements.
* **Complex Code Reasoning**: Measuring the model's deep analysis and reasoning skills regarding code logic, structure, and functionality.
* **Code Review Generation**: Evaluating the model's understanding of the logic behind a fix by generating a human-readable code review report.

SWE-CARE features two main task types:

1. **Issue Resolving**: Given a problem description (e.g., a GitHub issue), the model must generate a code patch to fix it. Evaluation is done by applying the patch and running tests in a reproducible environment.
2. **Code Review**: Given a code diff, the model must generate a comprehensive code review report. The quality of the report is assessed using a combination of automated metrics and LLM-as-a-judge evaluation.

The benchmark currently supports Python and Java.

## 🛠️ Set Up

Follow these steps to set up the project locally.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/SWE-CARE.git
    cd SWE-CARE
    ```

2. **Install dependencies:**
    This project uses `uv` for package management. Make sure you have Python 3.10 or higher.

    ```bash
    pip install uv
    uv sync
    ```

    Alternatively, you can use `pip`:

    ```bash
    pip install -e .
    ```

3. **Set up pre-commit hooks (for development):**
    This project uses `ruff` for linting and formatting. The pre-commit hooks will run these checks automatically before each commit.

    ```bash
    pre-commit install
    ```

## 📊 Data Collection

The data collection process involves several steps to gather and process data from GitHub. The main scripts for this process are located in `src/swe_care/collect`.

Here's an example of the command-line usage for each step:

1. **Get Top Repositories**: Find the most starred repositories for a given language.

    ```bash
    python -m swe_care.collect get_top_repos \
        --language "Python" \
        --top-n 100 \
        --output-dir "results/top_repos" \
        --tokens "your_github_pat"
    ```

2. **Get Pull Request Data**: Fetch PR data from a specific repository using the GitHub GraphQL API.

    ```bash
    python -m swe_care.collect get_graphql_prs_data \
        --repo "<repo_owner>/<repo_name>" \
        --output-dir "results/graphql_prs_data" \
        --tokens "your_github_pat" \
        --max-number 20
    ```

3. **Evaluate Commits**: Evaluate the collected commits from the PRs.

    **Single file processing:**

    ```bash
    python -m swe_care.collect evaluate_commits \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --output-dir "./results/evaluate_commits"
    ```

    **Batch processing (multiple repositories):**

    ```bash
    python -m swe_care.collect evaluate_commits \
        --graphql-prs-data-file "results/graphql_prs_data/" \
        --output-dir "./results/evaluate_commits" \
        --jobs 4
    ```

4. **Build Code Review Dataset**: Build the final dataset for the code review task.

    **Single file processing:**

    ```bash
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --pr-commits-evaluation-file "results/evaluate_commits/<repo_owner>__<repo_name>_pr_commits_evaluation.jsonl" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat"
    ```

    **Batch processing (multiple repositories):**

    ```bash
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/" \
        --pr-commits-evaluation-file "results/evaluate_commits/" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat" \
        --jobs 4
    ```

    **Note**: When using directory inputs, the tool will automatically:
    * Recursively find all `*_graphql_prs_data.jsonl` files in the specified directory
    * Match them with corresponding `*_pr_commits_evaluation.jsonl` files
    * Process multiple file pairs concurrently using the specified number of jobs

You can find more details about the arguments for each script by running `python -m swe_care.collect <subcommand> -h`.

## 🔄 Inference

The inference module provides two main functionalities: generating text datasets and running LLM inference on code review tasks.

### 1. Generate Text Datasets

Before running evaluation, you can generate text datasets from the collected SWE-CARE data with different context strategies. This creates datasets in the format required for LLM evaluation.

```bash
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "oracle" \
    --tokens "your_github_pat"
```

#### File Source Strategies

The `--file-source` parameter supports different strategies for selecting context files:

* **oracle**: Uses ground truth files (files that were actually changed in both the review commit and merged commit)
* **bm25**: Uses BM25 retrieval to select relevant files (requires `--retrieval-file`)
* **all**: Uses all available files up to a specified limit (requires `--k` parameter)

#### Additional Parameters for Text Generation

* `--k`: Maximum number of files to include (used with bm25 and all strategies)
* `--retrieval-file`: Path to BM25 retrieval results file (required for bm25 strategy)
* `--tokens`: GitHub Personal Access Token(s) for API access
* `--jobs`: Number of parallel jobs for multithreaded processing (default: 2)

### 2. Run LLM Inference

After generating text datasets, you can run inference using various LLM APIs to generate code review predictions.

```bash
# Example with OpenAI GPT-4o
export OPENAI_API_KEY=<your_openai_api_key>
python -m swe_care.inference run_api \
    --dataset-file "results/code_review_text/dataset__oracle.jsonl" \
    --model "gpt-4o" \
    --model-provider "openai" \
    --model-args "temperature=0.7,top_p=0.9" \
    --output-dir "results/predictions" \
    --jobs 4 \
    --skip-existing

# Example with Anthropic Claude
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
python -m swe_care.inference run_api \
    --dataset-file "results/code_review_text/dataset__oracle.jsonl" \
    --model "claude-3-5-sonnet-20241022" \
    --model-provider "anthropic" \
    --model-args "temperature=0.5,max_tokens=4096" \
    --output-dir "results/predictions" \
    --jobs 2

# Example with DeepSeek
export OPENAI_API_KEY=<your_deepseek_api_key>
python -m swe_care.inference run_api \
    --dataset-file "results/code_review_text/dataset__oracle.jsonl" \
    --model "deepseek-chat" \
    --model-provider "deepseek" \
    --output-dir "results/predictions" \
    --jobs 1
```

#### Supported Model Providers

See `python -m swe_care.inference run_api --help` for the supported model providers and models.

If you are using an API provider other than the provided ones, you can run inference with `export OPENAI_BASE_URL=<your_openai_base_url>` or `export ANTHROPIC_BASE_URL=<your_anthropic_base_url>` to specify the base URL for the API.

#### Parameters for LLM Inference

* `--dataset-file`: Path to the text dataset file (CodeReviewInferenceInstance objects)
* `--model`: Model name to use for inference
* `--model-provider`: Model provider (openai, anthropic, deepseek, qwen)
* `--model-args`: Comma-separated model arguments (e.g., `temperature=0.7,top_p=0.9`)
* `--output-dir`: Directory to save generated predictions
* `--jobs`: Number of parallel threads for inference (default: 2)
* `--skip-existing`: Skip instances that already have predictions (flag, default: False)

The generated predictions will be saved as JSONL files containing `CodeReviewPrediction` objects, which can then be used for evaluation.

You can find more details about the arguments for each script by running `python -m swe_care.inference <subcommand> -h`.

## 🚀 Evaluation

The evaluation harness is used to assess model predictions on the code review task. The main script is `src/swe_care/harness/code_review_eval.py`.

### Supported Evaluators and Model Providers for Evaluation

See `python -m swe_care.harness code_review_eval --help` for supported evaluators and LLM model if you want to use LLM-based evaluation.

### Examples

#### LLM-based Evaluation with OpenAI

```bash
export OPENAI_API_KEY=<your_openai_api_key>
python -m swe_care.harness code_review_eval \
    --dataset-file "results/code_review_task_instances.jsonl" \
    --predictions-path "results/predictions/dataset__gpt-4o.jsonl" \
    --output-dir "./results/evaluation" \
    --evaluator "llm_evaluator" \
    --model "gpt-4o" \
    --model-provider "openai" \
    --model-args "temperature=0.0" \
    --jobs 4
```

#### Rule-based Evaluation

```bash
python -m swe_care.harness code_review_eval \
    --dataset-file "results/code_review_task_instances.jsonl" \
    --predictions-path "results/predictions/dataset__gpt-4o.jsonl" \
    --output-dir "./results/evaluation" \
    --evaluator "rule_based_evaluator" \
    --jobs 4
```

#### Multiple Evaluators

```bash
export OPENAI_API_KEY=<your_openai_api_key>
python -m swe_care.harness code_review_eval \
    --dataset-file "results/code_review_task_instances.jsonl" \
    --predictions-path "results/predictions/dataset__gpt-4o.jsonl" \
    --output-dir "./results/evaluation" \
    --evaluator "llm_evaluator" "rule_based_evaluator" \
    --model "gpt-4o" \
    --model-provider "openai" \
    --jobs 4
```

### Parameters

* `--dataset-file`: Path to the original dataset file (CodeReviewTaskInstance objects)
* `--predictions-path`: Path to the predictions file (CodeReviewPrediction objects)
* `--output-dir`: Directory where evaluation results will be saved
* `--evaluator`: One or more evaluator types to use (`llm_evaluator`, `rule_based_evaluator`)
* `--model`: Model name for LLM evaluation (required if using `llm_evaluator`)
* `--model-provider`: Model provider for LLM evaluation (required if using `llm_evaluator`)
* `--model-args`: Comma-separated model arguments for LLM evaluation
* `--jobs`: Number of parallel threads for evaluation (default: 2)

### Output

The evaluation results are saved as a JSONL file (`final_report.jsonl`) containing `CodeReviewEvaluationResult` objects with detailed metrics for each instance.

You can find more details about the arguments for each script by running `python -m swe_care.harness <subcommand> -h`.

## 📜 Citation

(To be added)

## 🙏 Acknowledgements

(To be added)

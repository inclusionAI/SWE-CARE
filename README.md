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

## 🚀 Quick Start: Evaluation Pipeline

For a streamlined evaluation workflow, use the bootstrap script in `scripts/run_eval_pipeline.py`:

```bash
# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export LLM_EVALUATOR_OPENAI_API_KEY="your-o3-evaluation-api-key"

# Run the complete pipeline
python scripts/run_eval_pipeline.py \
    --dataset-file results/dataset/code_review_task_instances.jsonl \
    --output-dir results/pipeline_output \
    --model gpt-4o \
    --model-provider openai \
    --file-source oracle
```

This script automates the entire evaluation process: text generation → inference → evaluation. See [scripts/README.md](scripts/README.md) for detailed usage.

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

3. **Classify PRs Data**: Analyze and classify PR data by evaluating commits and labeling review comments.

    **Single file processing:**

    ```bash
    python -m swe_care.collect classify_prs_data \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --output-dir "./results/classify_prs_data" \
        --tokens "your_github_pat"
    ```

    **Batch processing (multiple repositories):**

    ```bash
    python -m swe_care.collect classify_prs_data \
        --graphql-prs-data-file "results/graphql_prs_data/" \
        --output-dir "./results/classify_prs_data" \
        --tokens "your_github_pat" \
        --jobs 4
    ```

    This step combines two important analyses:
    * **Commit Evaluation**: Uses heuristic rules to score commits based on quality indicators (message clarity, size, review activity, etc.)
    * **Review Comment Classification**: Extracts and labels review comments based on whether referenced lines were actually changed in the merged commit, or the review thread is resolved, outdated, or collapsed.

4. **Build Code Review Dataset**: Build the final dataset for the code review task. This step requires an LLM to classify metadata such as problem domain, difficulty, and review effort for each task instance.

    **Single file processing:**

    ```bash
    # Example with OpenAI GPT-4o
    export OPENAI_API_KEY=<your_openai_api_key>
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --pr-classification-file "results/classify_prs_data/<repo_owner>__<repo_name>_pr_classification.jsonl" \
        --model "gpt-4o" \
        --model-provider "openai" \
        --model-args "temperature=0.7,top_p=0.9" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat"

    # Example with Anthropic Claude
    export ANTHROPIC_API_KEY=<your_anthropic_api_key>
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
        --pr-classification-file "results/classify_prs_data/<repo_owner>__<repo_name>_pr_classification.jsonl" \
        --model "claude-3-5-sonnet-20241022" \
        --model-provider "anthropic" \
        --model-args "temperature=0.5,max_tokens=4096" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat"
    ```

    **Batch processing (multiple repositories):**

    ```bash
    export OPENAI_API_KEY=<your_openai_api_key>
    python -m swe_care.collect build_code_review_dataset \
        --graphql-prs-data-file "results/graphql_prs_data/" \
        --pr-classification-file "results/classify_prs_data/" \
        --model "gpt-4o" \
        --model-provider "openai" \
        --model-args "temperature=0.7" \
        --output-dir "./results/dataset" \
        --tokens "your_github_pat" \
        --jobs 4
    ```

    **Note**: When using directory inputs, the tool will automatically:
    * Recursively find all `*_graphql_prs_data.jsonl` files in the specified directory
    * Match them with corresponding `*_pr_classification.jsonl` files
    * Process multiple file pairs concurrently using the specified number of jobs

You can find more details about the arguments for each script by running `python -m swe_care.collect <subcommand> -h`.

### Convert to Reward Model Training Samples

This is an additional processing step that converts PR classification data to reward model training samples, separate from the main data collection pipeline.

**Single file processing:**

```bash
python -m swe_care.collect convert_to_rm_samples \
    --graphql-prs-data-file "results/graphql_prs_data/<repo_owner>__<repo_name>_graphql_prs_data.jsonl" \
    --pr-classification-file "results/classify_prs_data/<repo_owner>__<repo_name>_pr_classification.jsonl" \
    --output-dir "./results/rm_samples" \
    --file-source "none"
```

**Batch processing (multiple repositories):**

```bash
python -m swe_care.collect convert_to_rm_samples \
    --graphql-prs-data-file "results/graphql_prs_data/" \
    --pr-classification-file "results/classify_prs_data/" \
    --output-dir "./results/rm_samples" \
    --file-source "base_changed_files" \
    --jobs 4
```

**Using retrieval-based file sources:**

```bash
# Example with retrieved_all_files (requires --retrieval-output-dir)
python -m swe_care.collect convert_to_rm_samples \
    --graphql-prs-data-file "results/graphql_prs_data/" \
    --pr-classification-file "results/classify_prs_data/" \
    --output-dir "./results/rm_samples" \
    --file-source "retrieved_all_files" \
    --retrieval-output-dir "./results/retrieval_output" \
    --retrieval-max-files 10 \
    --jobs 2
```

This step converts classified PR data into training samples for reward models. Each sample contains:

* **Problem Statement**: Extracted from closing issues or PR description using the `extract_problem_statement` utility
* **Patch to Review**: The actual code changes (patch) from the commit
* **Positive Reviews**: Review comments where referenced lines were changed in the merged commit AND the review thread is resolved
* **Negative Reviews**: All other review comments
* **Metadata**: Repository info, PR number, commit SHA, PR URL, and file source for traceability

#### File Source Options

The `--file-source` parameter controls how file content is included in the review samples:

* **`none`** (default): Uses the default sample format without including changed files content
* **`base_changed_files`**: Includes the content of changed files from the patch between base commit and commit to review in the review comment sample
* **`reviewed_file`**: Includes changed file content to the sample the review comment applied to
* **`retrieved_base_changed_files`**: Uses BM25 to retrieve relevant files from changed files based on the diff_hunk content
* **`retrieved_all_files`**: Uses BM25 to retrieve relevant files from the entire repository based on the diff_hunk content

When `--file-source` is set to any option other than `none`, review comments will include a `<code>` section containing the relevant file content, providing more context for training. The retrieval-based options (`retrieved_base_changed_files` and `retrieved_all_files`) use BM25 similarity to select the most relevant files based on the review comment's diff_hunk.

**Note**: When using `--file-source retrieved_all_files`, you must also specify `--retrieval-output-dir` to set the directory where retrieval operations will be performed and temporary files will be stored.

**Important**: The `retrieved_all_files` file source strategy uses Pyserini for BM25 retrieval, which requires Java 21. Make sure Java 21 is installed on your system before using this option. See [Pyserini installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md) for details.

The output files follow the naming pattern `<repo_owner>__<repo_name>_rm_samples.jsonl` and contain `RewardModelTrainingSample` objects with comprehensive metadata for each training instance.

## 🔄 Inference

The inference module provides two main functionalities: generating text datasets and running LLM inference on code review tasks.

### 1. Generate Text Datasets

Before running evaluation, you can generate text datasets from the collected SWE-CARE data with different context strategies. This creates datasets in the format required for LLM evaluation.

```bash
# Example with no file context
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "none"

# Example with oracle file source
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "oracle" \
    --tokens "your_github_pat"

# Example with BM25 retrieval
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "bm25" \
    --k 10 \
    --retrieval-output-dir "results/retrieval_output" \
    --tokens "your_github_pat" \
    --jobs 4

# Example with all files
python -m swe_care.inference create_code_review_text \
    --dataset-file "results/dataset/code_review_task_instances.jsonl" \
    --output-dir "results/code_review_text" \
    --file-source "all" \
    --k 20 \
    --retrieval-output-dir "results/retrieval_output" \
    --tokens "your_github_pat" \
    --jobs 4
```

#### File Source Strategies

The `--file-source` parameter supports different strategies for selecting context files:

* **none**: No file context, only uses problem statement and patch
* **oracle**: Uses ground truth files (files that were actually changed in both the review commit and merged commit)
* **bm25**: Uses BM25 retrieval to select relevant files based on the problem statement (requires `--k` and `--retrieval-output-dir`)
* **all**: Uses all available files from the repository up to a specified limit (requires `--k` and `--retrieval-output-dir`)

#### Additional Parameters for Text Generation

* `--k`: Maximum number of files to include (required for bm25 and all strategies)
* `--retrieval-output-dir`: Directory for retrieval operations and git repositories (required for bm25 and all strategies)
* `--tokens`: GitHub Personal Access Token(s) for API access
* `--jobs`: Number of parallel jobs for processing (default: 2). Uses ProcessPoolExecutor for bm25/all strategies for better parallelism
* `--skip-existing`: Skip existing instances in the output file based on instance_id (default: False). When specified, the tool will append to the existing output file instead of overwriting it

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
* `--model-provider`: Model provider (openai, anthropic, deepseek, qwen, moonshot, gemini)
* `--model-args`: Comma-separated model arguments (e.g., `temperature=0.7,top_p=0.9`)
* `--output-dir`: Directory to save generated predictions
* `--jobs`: Number of parallel threads for inference (default: 2)
* `--skip-existing`: Skip instances that already have predictions (flag, default: False)

The generated predictions will be saved as JSONL files containing `CodeReviewPrediction` objects, which can then be used for evaluation.

You can find more details about the arguments for each script by running `python -m swe_care.inference <subcommand> -h`.

## 🚀 Evaluation

The evaluation harness is used to assess model predictions on the code review task. The main script is `src/swe_care/harness/code_review_eval.py`.

### Supported Evaluators

1. **LLM Evaluator** (`llm_evaluator`): Evaluates code review quality based on multiple dimensions (functionality, quality, style, documentation).
2. **Rule-based Evaluator** (`rule_based_evaluator`): Extracts defects from review text and compares them with reference defects.
3. **Repo-level LLM Evaluator** (`repo_level_llm_evaluator`): Uses LLM to classify review comments as positive or negative based on problem statement and patch context.

### Supported Model Providers for Evaluation

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

#### Repo-level LLM Evaluation with File Context

```bash
export OPENAI_API_KEY=<your_openai_api_key>
python -m swe_care.harness code_review_eval \
    --dataset-file "results/code_review_task_instances.jsonl" \
    --predictions-path "results/predictions/dataset__gpt-4o.jsonl" \
    --output-dir "./results/evaluation" \
    --evaluator "repo_level_llm_evaluator" \
    --model "gpt-4o" \
    --model-provider "openai" \
    --evaluator-args "repo_level_llm_evaluator:file_source=retrieved_all_files,retrieval_max_files=10,retrieval_output_dir=./results/retrieval_output" \
    --jobs 4
```

### Parameters

* `--dataset-file`: Path to the original dataset file (CodeReviewTaskInstance objects)
* `--predictions-path`: Path to the predictions file (CodeReviewPrediction objects)
* `--output-dir`: Directory where evaluation results will be saved
* `--evaluator`: One or more evaluator types to use (`llm_evaluator`, `rule_based_evaluator`, `repo_level_llm_evaluator`)
* `--model`: Model name for LLM evaluation (required if using LLM-based evaluators)
* `--model-provider`: Model provider for LLM evaluation (required if using LLM-based evaluators)
* `--model-args`: Comma-separated model arguments for LLM evaluation
* `--evaluator-args`: Evaluator-specific arguments in format `evaluator1:arg1=value1,arg2=value2;evaluator2:arg1=value1`
* `--jobs`: Number of parallel threads for evaluation (default: 2)

#### Evaluator-specific Arguments

For `repo_level_llm_evaluator`:

* `file_source`: Source for file content (`none`, `base_changed_files`, `reviewed_file`, `retrieved_base_changed_files`, `retrieved_all_files`)
* `retrieval_max_files`: Maximum number of files for retrieval (default: 5)
* `retrieval_output_dir`: Output directory for retrieval operations (required when `file_source` is `retrieved_all_files`)
* `tokens`: GitHub API tokens (passed automatically from global tokens)

### Output

The evaluation results are saved as a JSONL file (`final_report.jsonl`) containing `CodeReviewEvaluationResult` objects with detailed metrics for each instance.

You can find more details about the arguments for each script by running `python -m swe_care.harness <subcommand> -h`.

## 📜 Citation

(To be added)

## 🙏 Acknowledgements

(To be added)

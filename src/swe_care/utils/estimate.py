from typing import Any

from loguru import logger

from swe_care.utils.llm_models.clients import BaseModelClient


def classify_problem_domain(client: BaseModelClient, problem_statement: str) -> str:
    """Classify problem domain of a code review task."""
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

    response = client.create_completion(messages)
    return response.strip()


def estimate_difficulty(
    client: BaseModelClient, pr_data: dict[str, Any], commit_message: str, patch: str
) -> str:
    """Estimate the implementation difficulty for a pull request task."""

    title = pr_data.get("title", "")
    body = pr_data.get("body", "")
    changed_files = pr_data.get("changedFiles", "")

    system_prompt = """You are an experienced software engineer responsible for estimating implementation difficulty. Your task is to estimate how difficult it would be to implement a pull request from scratch on a scale of 1 to 3, where:

1 = Low Difficulty: Simple implementations like typo fixes, minor configuration changes, straightforward bug fixes with clear solutions, basic feature additions using existing patterns, or routine maintenance tasks.

2 = Medium Difficulty: Moderate implementations requiring some problem-solving, such as bug fixes requiring investigation, feature additions involving multiple components, refactoring existing code, integration with existing APIs, or changes requiring understanding of business logic.

3 = High Difficulty: Complex implementations requiring significant technical expertise, such as architectural changes, performance optimizations, complex algorithms, new integrations with external systems, security-related fixes, or features requiring deep domain knowledge.

Consider factors like:
- Technical complexity of the problem being solved
- Amount of new code vs. modifications to existing code
- Number of systems/components involved
- Domain knowledge required
- Algorithm complexity
- Architectural impact
- Dependencies and integrations needed
- Research or investigation required

Please respond with ONLY a single number from 1 to 3."""

    user_prompt = f"""Please estimate the implementation difficulty level (1, 2, 3) for the following pull request:

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

Implementation Difficulty (1, 2, 3):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.create_completion(messages)
    difficulty = int(response.strip())
    # Ensure the difficulty is within valid range
    difficulty = max(1, min(3, difficulty))

    if difficulty == 1:
        return "low"
    elif difficulty == 2:
        return "medium"
    elif difficulty == 3:
        return "high"


def classify_review_effort(
    client: BaseModelClient, pr_data: dict[str, Any], commit_message: str, patch: str
) -> int:
    """Estimate review effort for a code review task."""
    labels = pr_data.get("labels", [])
    if "nodes" in labels:
        for node in labels["nodes"]:
            if "Review effort" in node.get("name"):
                try:
                    effort = node.get("name").split("Review effort")[-1].strip()
                    if "/" in effort:
                        effort = effort.split("/")[0]
                    elif ":" in effort:
                        effort = effort.split(":")[-1].strip()
                    return int(effort)
                except Exception as e:
                    logger.warning(
                        f"Error parsing review effort, fallback to LLM estimate: {e}"
                    )
                    break

    title = pr_data.get("title", "")
    body = pr_data.get("body", "")

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

    response = client.create_completion(messages)
    effort = int(response.strip())
    # Ensure the effort is within valid range
    return max(1, min(5, effort))

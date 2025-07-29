from typing import Any
from swe_care.utils.classify_code_review_tasks import (classify_problem_domain, estimate_review_effort,estimate_review_difficulty)
from swe_care.utils.llm_models import init_llm_client

def estimate_problem_domains(pr_data: dict[str, Any], problem_statement: str) -> list[str]:
    """Estimate problem domains based on PR data. For now, returns empty list."""
    # TODO: Implement domain estimation logic
    client = init_llm_client(
            model="gemini-2.5-pro-preview-06-05",
            model_provider="openai",
            temperature=0.1,
        )
    problem_domains = classify_problem_domain(client,problem_statement)
    return [problem_domains]


def estimate_difficulty(pr_data: dict[str, Any], commit_message: str, patch: str) -> str:
    """Estimate difficulty based on PR data. For now, returns default."""
    # TODO: Implement difficulty estimation logic
    client = init_llm_client(
            model="gemini-2.5-pro-preview-06-05",
            model_provider="openai",
            temperature=0.1,
        )
    diffuculty = estimate_review_difficulty(
        client,
        pr_data.get("title", ''),
        pr_data.get("body", ''),
        commit_message,
        pr_data.get("changedFiles", ''),
        patch
    )
    if diffuculty == 1:
        return "low"
    elif diffuculty == 2:
        return "medium"
    elif diffuculty == 3:
        return "high"

def classify_review_effort(pr_data: dict[str, Any], commit_message: str, patch: str) -> int:
    """Estimate review effort based on PR data. For now, returns default."""
    # TODO: Implement review effort estimation logic
    labels = pr_data.get("labels", [])
    if "nodes" in labels:
        for node in labels["nodes"]:
            if "Review effort" in node.get("name"):
                return int(node.get("name").split("Review effort")[-1].strip().split('/')[0])
    client = init_llm_client(
            model="gemini-2.5-pro-preview-06-05",
            model_provider="openai",
            temperature=0.1,
        )
    return estimate_review_effort(client, pr_data.get("title", ''), pr_data.get("body", ''), commit_message, patch)

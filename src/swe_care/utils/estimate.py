from typing import Any


def estimate_problem_domains(pr_data: dict[str, Any]) -> list[str]:
    """Estimate problem domains based on PR data. For now, returns empty list."""
    # TODO: Implement domain estimation logic
    return []


def estimate_difficulty(pr_data: dict[str, Any]) -> str:
    """Estimate difficulty based on PR data. For now, returns default."""
    # TODO: Implement difficulty estimation logic
    return "medium"


def estimate_review_effort(pr_data: dict[str, Any]) -> int:
    """Estimate review effort based on PR data. For now, returns default."""
    # TODO: Implement review effort estimation logic
    return 3

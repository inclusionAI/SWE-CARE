from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ResolvedIssue:
    """Schema for resolved issue instances."""

    number: int
    title: str
    body: str


@dataclass_json
@dataclass
class Metadata:
    """Schema for metadata instances."""

    problem_domains: list[str]
    difficulty: str


@dataclass_json
@dataclass
class IssueResolvingTask:
    """Schema for Issue Resolving task instances."""

    instance_id: str
    repo: str
    language: str
    pull_number: int
    title: str
    body: str
    created_at: str
    problem_statement: str
    hints_text: str
    resolved_issues: list[ResolvedIssue]
    base_commit: str
    patch: str
    test_patch: str
    env_setup_config: dict[str, Any]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    version: str
    metadata: Metadata


@dataclass_json
@dataclass
class CodeReviewTask:
    """Schema for Code Review task instances."""

    instance_id: str
    repo: str
    language: str
    pull_number: int
    title: str
    body: str
    created_at: str
    problem_statement: str
    hints_text: str
    resolved_issues: list[ResolvedIssue]
    base_commit: str
    review_head_commit: str
    review_head_commit_message: str
    review_commit_known_issues: list[dict[str, Any]]
    patch: str
    metadata: Metadata

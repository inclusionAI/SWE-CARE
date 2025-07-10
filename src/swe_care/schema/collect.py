from dataclasses import dataclass

from dataclasses_json import dataclass_json

from swe_care.schema.dataset import ReferenceReviewComment


@dataclass_json
@dataclass
class ReviewCommentLabels:
    """Labels for a review comment."""

    referenced_line_changed_in_merged_commit: bool
    """Whether the referenced line was changed in the merged commit"""
    is_resolved: bool
    """Whether the review thread was resolved"""
    is_outdated: bool
    """Whether the review thread is outdated"""
    is_collapsed: bool
    """Whether the review thread is collapsed"""


@dataclass_json
@dataclass
class LabeledReviewComment(ReferenceReviewComment):
    """Schema for labeled review comment instances."""

    labels: ReviewCommentLabels
    """Labels for the review comment"""


@dataclass_json
@dataclass
class CommitWithLabeledReviewComments:
    """Schema for commit with labeled review comments."""

    commit_sha: str
    """The commit SHA"""
    labeled_review_comments: list[LabeledReviewComment]
    """List of labeled review comments for this commit"""


@dataclass_json
@dataclass
class PRCommitWithLabeledReviewComments:
    """Schema for PR commit with labeled review comments."""

    repo_owner: str
    """Repository owner"""
    repo_name: str
    """Repository name"""
    pr_number: int
    """Pull request number"""
    url: str
    """Pull request URL"""
    commit_with_labeled_review_comments: list[CommitWithLabeledReviewComments]
    """List of commits with their labeled review comments"""


@dataclass_json
@dataclass
class CommitEvaluationResult:
    """Results from evaluating a commit."""

    commit_sha: str
    total_score: float
    rule_results: dict[str, bool | float]
    rule_category: dict[str, str]


@dataclass_json
@dataclass
class PRCommitEvaluation:
    """Results from evaluating a PR."""

    repo_owner: str
    repo_name: str
    pr_number: int
    url: str
    commits: list[CommitEvaluationResult]

from dataclasses import dataclass

from dataclasses_json import dataclass_json

from swe_care.schema.dataset import ReferenceReviewComment


@dataclass_json
@dataclass
class ReviewCommentLabels:
    """Labels for a review comment."""

    referenced_line_changed_in_merged_commit: bool
    """Whether the referenced line was changed in the merged commit. If True, the review comment was more likely to address real issues that got fixed."""
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
class CommitClassificationResult:
    """Schema combining commit evaluation and labeled review comments."""

    commit_sha: str
    """The commit SHA"""
    labeled_review_comments: list[LabeledReviewComment]
    """List of labeled review comments for this commit"""
    total_score: float
    """Total evaluation score for the commit"""
    rule_results: dict[str, bool | float]
    """Results from evaluation rules"""


@dataclass_json
@dataclass
class PRClassification:
    """Schema for PR with combined commit classification data."""

    repo_owner: str
    """Repository owner"""
    repo_name: str
    """Repository name"""
    pr_number: int
    """Pull request number"""
    url: str
    """Pull request URL"""
    commits: list[CommitClassificationResult]
    """List of commits with classification data (evaluation + labeled review comments)"""

"""
Utility functions for retrieving relevant files based on different strategies.
"""

import string
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from swe_care.schema.collect import LabeledReviewComment
from swe_care.schema.dataset import ReferenceReviewComment
from swe_care.utils.extract_prs_data import (
    fetch_repo_file_content,
    fetch_repo_files_content_by_retrieval,
)
from swe_care.utils.patch import get_changed_file_paths


def get_relevant_files(
    review_comment: ReferenceReviewComment | LabeledReviewComment,
    file_source: Literal[
        "none",
        "base_changed_files",
        "reviewed_file",
        "retrieved_base_changed_files",
        "retrieved_all_files",
    ],
    repo: str,
    base_commit: str,
    patch_to_review: str,
    tokens: Optional[list[str]] = None,
    retrieval_max_files: int = 5,
    retrieval_output_dir: Optional[Path | str] = None,
    changed_files: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Get relevant files based on file_source strategy.

    Args:
        review_comment: The review comment (can be ReferenceReviewComment or LabeledReviewComment)
        file_source: Source for file content
        repo: Repository in format 'owner/name'
        base_commit: Base commit SHA
        patch_to_review: Patch content to review
        tokens: GitHub API tokens
        retrieval_max_files: Maximum number of files for retrieval
        retrieval_output_dir: Output directory for retrieval operations
        changed_files: Pre-computed changed files (optional, to avoid re-fetching)

    Returns:
        Dictionary mapping file paths to their contents
    """
    if file_source == "none":
        return {}

    if file_source == "reviewed_file" and review_comment.path:
        try:
            content = fetch_repo_file_content(
                repo, base_commit, review_comment.path, tokens, patch_to_review
            )
            return {review_comment.path: content}
        except Exception as e:
            logger.warning(f"Failed to fetch content for {review_comment.path}: {e}")
            return {}

    elif file_source == "base_changed_files":
        if changed_files is not None:
            return changed_files
        else:
            return get_changed_files_in_patch(
                repo, base_commit, patch_to_review, tokens
            )

    elif file_source == "retrieved_base_changed_files":
        if not review_comment.diff_hunk:
            return {}

        # Get changed files first
        if changed_files is None:
            changed_files = get_changed_files_in_patch(
                repo, base_commit, patch_to_review, tokens
            )

        if not changed_files:
            return {}

        # Use BM25 to retrieve relevant files
        def preprocess(text):
            text = text.lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            return word_tokenize(text)

        try:
            query_tokens = preprocess(review_comment.diff_hunk)
            bm25 = BM25Okapi(
                [preprocess(content) for content in changed_files.values()]
            )
            doc_scores = bm25.get_scores(query_tokens)

            top_indices = sorted(
                range(len(doc_scores)),
                key=lambda i: doc_scores[i],
                reverse=True,
            )[: min(retrieval_max_files, len(changed_files))]

            file_paths = list(changed_files.keys())
            relevant_files = {}
            for idx in top_indices:
                file_path = file_paths[idx]
                relevant_files[file_path] = changed_files[file_path]
            return relevant_files
        except Exception as e:
            logger.warning(f"Failed to retrieve content for diff hunk: {e}")
            return {}

    elif file_source == "retrieved_all_files":
        if not review_comment.diff_hunk:
            return {}

        if retrieval_output_dir is None:
            raise ValueError(
                "retrieval_output_dir is required when file_source is 'retrieved_all_files'"
            )

        # Convert Path to str if needed
        if isinstance(retrieval_output_dir, Path):
            retrieval_output_dir = str(retrieval_output_dir)

        return fetch_repo_files_content_by_retrieval(
            repo=repo,
            commit=base_commit,
            query=review_comment.diff_hunk,
            retrieval_output_dir=retrieval_output_dir,
            tokens=tokens,
            max_files=retrieval_max_files,
        )

    return {}


def get_changed_files_in_patch(
    repo: str,
    base_commit: str,
    patch_to_review: str,
    tokens: list[str] | None = None,
) -> dict[str, str]:
    """
    Get file path and file content that are changed in the patch.
    """
    changed_files = {}

    logger.debug(f"Getting changed file paths from {base_commit} to commit_to_review")
    changed_file_paths = get_changed_file_paths(patch_to_review)
    logger.debug(f"Changed file paths: {changed_file_paths}")

    # Fetch file contents
    for file_path in changed_file_paths:
        try:
            logger.debug(f"Fetching content for {file_path}")
            content = fetch_repo_file_content(
                repo, base_commit, file_path, tokens, patch_to_review
            )
            changed_files[file_path] = content
        except Exception as e:
            logger.warning(f"Failed to fetch content for {file_path}: {e}")
            changed_files[file_path] = ""

    # Filter out files without content and return only the files we fetched
    result = {
        path: content for path, content in changed_files.items() if content is not None
    }

    logger.info(f"Retrieved {len(result)} changed files")
    return result

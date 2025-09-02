"""
Utility functions for evaluators.
"""

import re
from typing import Optional

import unidiff

from swe_care.schema.dataset import (
    CodeReviewTaskInstance,
    ReferenceReviewComment,
)


def _parse_diff_hunks(diff_text: str) -> list[dict]:
    """Parse diff hunks from a patch text."""
    patch = unidiff.PatchSet(diff_text)
    results = []

    for patched_file in patch:
        old_path = patched_file.path
        for hunk in patched_file:
            hunk_info = {
                "file": old_path,
                "old_start": hunk.source_start,
                "old_lines": hunk.source_length,
                "old_end": hunk.source_start + hunk.source_length - 1,
                "new_start": hunk.target_start,
                "new_lines": hunk.target_length,
                "new_end": hunk.target_start + hunk.target_length - 1,
                "hunk": str(hunk),
            }
            results.append(hunk_info)
    return results


def extract_defects_from_review(
    review_text: str, input: Optional[CodeReviewTaskInstance] = None
) -> list[ReferenceReviewComment]:
    """Extract defects from review text that follows the REVIEW_PROMPT format.

    Args:
        review_text: The code review text containing defects in <defect> tags
        input: Optional input task instance for extracting diff hunks

    Returns:
        List of ReferenceReviewComment objects extracted from the review text
    """
    defects = []

    # Pattern to match <defect> blocks
    defect_pattern = r"<defect>\s*(.*?)\s*</defect>"
    defect_matches = re.findall(defect_pattern, review_text, re.DOTALL)

    for defect_content in defect_matches:
        # Extract file_path, line, and suggestion from defect content
        file_path_match = re.search(r"file_path:\s*(.+)", defect_content)
        line_match = re.search(r"line:\s*(\d+)", defect_content)
        suggestion_match = re.search(r"suggestion:\s*(.+)", defect_content, re.DOTALL)

        if file_path_match and suggestion_match:
            file_path = file_path_match.group(1).strip()
            line_num = int(line_match.group(1)) if line_match else None
            suggestion = suggestion_match.group(1).strip()

            # Extract diff hunk based on patch_to_review, file path line number
            # if the line number falls within a hunk, use that hunk.
            patch = (
                input.commit_to_review.patch_to_review
                if input and input.commit_to_review
                else None
            )
            diff_hunks = _parse_diff_hunks(patch) if patch else []
            diff_hunk = None

            for hunk in diff_hunks:
                if (
                    hunk["file"] == file_path
                    and line_num is not None
                    and hunk["new_start"] <= line_num <= hunk["new_end"]
                ):
                    diff_hunk = hunk["hunk"]
                    break

            defect = ReferenceReviewComment(
                text=suggestion,
                path=file_path,
                diff_hunk=diff_hunk,
                line=line_num,
                start_line=None,
                original_line=None,
                original_start_line=None,
            )
            defects.append(defect)

    return defects

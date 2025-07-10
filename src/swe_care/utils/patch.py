import unidiff
from loguru import logger


def get_changed_file_paths(patch_content: str) -> list[str]:
    """
    Extract file paths that are changed in a patch.

    Args:
        patch_content: The patch content as a string

    Returns:
        A list of file paths that are modified in the patch
    """
    try:
        patch_set = unidiff.PatchSet(patch_content)
        changed_files = []

        for patched_file in patch_set:
            file_path = patched_file.source_file
            if file_path.startswith("a/"):
                file_path = file_path[2:]  # Remove 'a/' prefix

            if file_path == "/dev/null":
                logger.debug("Skipping /dev/null file, this is a new file")
                continue

            changed_files.append(file_path)

        return changed_files
    except Exception:
        # If parsing fails, return empty list
        return []


def get_changed_lines_in_file(patch_content: str, file_path: str) -> set[int]:
    """
    Extract line numbers that were changed in a specific file from a patch.

    Args:
        patch_content: The patch content as a string
        file_path: The file path to analyze

    Returns:
        A set of line numbers that were modified in the file
    """
    try:
        patch_set = unidiff.PatchSet(patch_content)
        changed_lines = set()

        for patched_file in patch_set:
            # Check if this is the file we're interested in
            source_file = patched_file.source_file
            target_file = patched_file.target_file

            # Remove 'a/' and 'b/' prefixes
            if source_file.startswith("a/"):
                source_file = source_file[2:]
            if target_file.startswith("b/"):
                target_file = target_file[2:]

            if source_file == file_path or target_file == file_path:
                # Analyze hunks to find changed lines
                for hunk in patched_file:
                    for line in hunk:
                        if line.is_added or line.is_removed:
                            # For removed lines, use source line number
                            # For added lines, use target line number
                            if line.is_removed and line.source_line_no:
                                changed_lines.add(line.source_line_no)
                            elif line.is_added and line.target_line_no:
                                changed_lines.add(line.target_line_no)

        return changed_lines
    except Exception as e:
        logger.warning(f"Failed to parse patch for file {file_path}: {e}")
        return set()


def is_line_changed_in_patch(
    patch_content: str, file_path: str, line_number: int
) -> bool:
    """
    Check if a specific line was changed in the patch.

    Args:
        patch_content: The patch content as a string
        file_path: The file path to check
        line_number: The line number to check

    Returns:
        True if the line was changed, False otherwise
    """
    changed_lines = get_changed_lines_in_file(patch_content, file_path)
    return line_number in changed_lines

from typing import Any

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from swe_care.schema.dataset import ReferenceReviewComment
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.prompt_loader import load_prompt


class InvalidResponseError(Exception):
    """Raised when LLM returns an invalid response that should trigger a retry."""

    pass


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(InvalidResponseError),
    reraise=True,
)
def classify_problem_domain(client: BaseModelClient, problem_statement: str) -> str:
    """Classify problem domain of a code review task."""
    valid_categories = {
        "Bug Fixes",
        "New Feature Additions",
        "Code Refactoring / Architectural Improvement",
        "Documentation Updates",
        "Test Suite / CI Enhancements",
        "Performance Optimizations",
        "Security Patches / Vulnerability Fixes",
        "Dependency Updates & Env Compatibility",
        "Code Style, Linting, Formatting Fixes",
    }

    system_prompt, user_prompt = load_prompt(
        "classify_problem_domain", problem_statement=problem_statement
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages).strip()
        if response in valid_categories:
            return response
        else:
            logger.warning(f"Invalid category response: '{response}'")
            raise InvalidResponseError(f"Invalid response: '{response}'")
    except Exception as e:
        if isinstance(e, InvalidResponseError):
            raise
        logger.warning(f"Error in classify_problem_domain: {e}")
        raise InvalidResponseError(f"LLM call failed: {e}")


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(InvalidResponseError),
    reraise=True,
)
def estimate_difficulty(
    client: BaseModelClient, pr_data: dict[str, Any], commit_message: str, patch: str
) -> str:
    """Estimate the implementation difficulty for a pull request task."""

    title = pr_data.get("title", "")
    body = pr_data.get("body", "")
    changed_files = pr_data.get("changedFiles", "")

    system_prompt, user_prompt = load_prompt(
        "estimate_difficulty",
        title=title,
        body=body,
        commit_message=commit_message,
        changed_files=str(changed_files),
        patch=patch,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages).strip()
        difficulty = int(response)
        if difficulty in [1, 2, 3]:
            if difficulty == 1:
                return "low"
            elif difficulty == 2:
                return "medium"
            elif difficulty == 3:
                return "high"
        else:
            logger.warning(f"Invalid difficulty response: '{response}'")
            raise InvalidResponseError(
                f"Invalid response: '{response}' (expected 1, 2, or 3)"
            )
    except ValueError:
        logger.warning(f"Could not parse difficulty response: {response}")
        raise InvalidResponseError(f"Could not parse response: '{response}'")
    except Exception as e:
        if isinstance(e, InvalidResponseError):
            raise
        logger.warning(f"Error in estimate_difficulty: {e}")
        raise InvalidResponseError(f"LLM call failed: {e}")


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

    system_prompt, user_prompt = load_prompt(
        "classify_review_effort",
        title=title,
        body=body,
        commit_message=commit_message,
        patch=patch,
    )

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(InvalidResponseError),
        reraise=True,
    )
    def _llm_call():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.create_completion(messages).strip()
            effort = int(response)
            if effort in [1, 2, 3, 4, 5]:
                return effort
            else:
                logger.warning(f"Invalid effort response: '{response}'")
                raise InvalidResponseError(
                    f"Invalid response: '{response}' (expected 1-5)"
                )
        except ValueError:
            logger.warning(f"Could not parse effort response: {response}")
            raise InvalidResponseError(f"Could not parse response: '{response}'")
        except Exception as e:
            if isinstance(e, InvalidResponseError):
                raise
            logger.warning(f"Error in classify_review_effort: {e}")
            raise InvalidResponseError(f"LLM call failed: {e}")

    try:
        return _llm_call()
    except InvalidResponseError:
        raise ValueError(
            "Failed to get valid review effort estimate (1-5) after 3 attempts"
        )


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(InvalidResponseError),
    reraise=True,
)
def classify_relevant_review_comment(
    client: BaseModelClient, review_comment: ReferenceReviewComment
) -> bool:
    """Classify if a review comment is relevant (likely to lead to code changes).

    Args:
        client: The LLM client to use for classification
        review_comment: The review comment to classify

    Returns:
        True if the comment is relevant (likely to lead to code changes),
        False if irrelevant (unlikely to lead to code changes)
    """

    # Determine the line number to check for line change detection
    line_to_check = None
    if review_comment.original_line is not None:
        line_to_check = review_comment.original_line
    elif review_comment.line is not None:
        line_to_check = review_comment.line
    elif review_comment.start_line is not None:
        line_to_check = review_comment.start_line
    elif review_comment.original_start_line is not None:
        line_to_check = review_comment.original_start_line

    system_prompt, user_prompt = load_prompt(
        "classify_relevant_review_comment",
        comment_text=review_comment.text,
        diff_hunk=review_comment.diff_hunk,
        path=review_comment.path,
        line=line_to_check,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.create_completion(messages).strip().lower()
        if response == "relevant":
            return True
        elif response == "irrelevant":
            return False
        else:
            logger.warning(
                f"Invalid relevance classification response: '{response}' "
                f"(expected 'relevant' or 'irrelevant')"
            )
            raise InvalidResponseError(
                f"Invalid response: '{response}' (expected 'relevant' or 'irrelevant')"
            )
    except Exception as e:
        if isinstance(e, InvalidResponseError):
            raise
        logger.warning(f"Error in classify_relevant_review_comment: {e}")
        raise InvalidResponseError(f"LLM call failed: {e}")

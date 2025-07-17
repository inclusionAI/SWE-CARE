import json
import random
import time
import urllib.parse
from typing import Any, Optional

import requests
from loguru import logger
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class MaxNodeLimitExceededError(Exception):
    """Exception raised when the maximum node limit is exceeded."""

    def __init__(self, error: dict):
        self.error = error
        super().__init__(f"Max node limit exceeded: {error}")


class GitHubAPI:
    """GitHub API client with rate limiting, retries, and proper error handling."""

    def __init__(
        self,
        max_retries: int = 5,
        timeout: int = 60,
        tokens: Optional[list[str]] = None,
    ):
        """
        Initialize GitHub API client.

        Args:
            max_retries: Maximum number of retries for API calls
            tokens: Optional list of GitHub tokens for authentication
        """
        self.max_retries = max_retries
        self.tokens = tokens or []
        self.timeout = timeout
        self.graphql_endpoint = "https://api.github.com/graphql"
        self.rest_api_base = "https://api.github.com"

    def _get_token(self) -> Optional[str]:
        """Get a randomly selected token from the available tokens."""
        if not self.tokens:
            return None
        return random.choice(self.tokens)

    def _get_headers(self, content_type: str = "application/json") -> dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": content_type,
            "Accept": "application/vnd.github+json",
        }

        token = self._get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        return headers

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle rate limiting based on response headers."""
        if "X-RateLimit-Remaining" in response.headers:
            remaining = int(response.headers["X-RateLimit-Remaining"])
            if remaining < 10:  # If less than 10 requests remaining
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                current_time = int(time.time())
                wait_time = max(0, reset_time - current_time)
                if wait_time > 0:
                    logger.info(
                        f"Rate limit approaching. Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

    def _retry_wrapper(self, func):
        """Create a retry wrapper using the instance's max_retries setting."""
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=4, max=60),
            retry=retry_if_exception_type(
                (
                    requests.exceptions.RequestException,
                    requests.exceptions.HTTPError,
                )
            ),
            before_sleep=before_sleep_log(logger, "WARNING"),
            after=after_log(logger, "WARNING"),
        )(func)

    def execute_graphql_query(
        self, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query with retry mechanism.

        Args:
            query: GraphQL query string
            variables: Variables for the GraphQL query

        Returns:
            JSON response from the GraphQL API

        Raises:
            ValueError: If GraphQL errors are returned
            requests.exceptions.RequestException: If the request fails
        """

        def _execute_query():
            headers = self._get_headers()
            payload = {
                "query": query,
                "variables": variables,
            }

            response = requests.post(
                self.graphql_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Handle rate limiting
            self._handle_rate_limit(response)

            results = response.json()

            # Check for GraphQL errors
            if "errors" in results:
                # Check if it's a node limit error that we can handle
                for error in results["errors"]:
                    if error.get("type") == "MAX_NODE_LIMIT_EXCEEDED":
                        raise MaxNodeLimitExceededError(error)

                raise ValueError(f"GraphQL errors: {results['errors']}")

            return results

        return self._retry_wrapper(_execute_query)()

    def call_api(
        self,
        url: str,
        method: str = "GET",
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Call GitHub REST API endpoint with retry mechanism.

        Args:
            url: Full URL or path relative to GitHub API base
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            timeout: Request timeout in seconds

        Returns:
            Response object

        Raises:
            requests.exceptions.RequestException: If the request fails
        """

        def _call_api():
            # Handle both full URLs and relative paths
            full_url = url
            if not url.startswith("http"):
                full_url = f"{self.rest_api_base}/{url.lstrip('/')}"

            headers = self._get_headers()

            response = requests.request(
                method=method,
                url=full_url,
                headers=headers,
                params=params,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Handle rate limiting
            self._handle_rate_limit(response)

            return response

        return self._retry_wrapper(_call_api)()

    def get_patch(
        self,
        repo: str,
        *,
        pr_number: Optional[int] = None,
        base_commit: Optional[str] = None,
        head_commit: Optional[str] = None,
    ) -> str:
        """
        Get patch/diff for a PR or commit range.

        Args:
            repo: Repository in format 'owner/repo'
            pr_number: Pull request number (for PR patch)
            base_commit: Base commit for commit range
            head_commit: Head commit for commit range

        Returns:
            Patch/diff content as string

        Raises:
            ValueError: If neither pr_number nor commit range is provided
            requests.exceptions.RequestException: If the request fails
        """

        def _get_patch():
            if pr_number is not None:
                patch_url = f"https://github.com/{repo}/pull/{pr_number}.diff"
            elif base_commit and head_commit:
                patch_url = f"https://github.com/{repo}/compare/{base_commit}...{head_commit}.diff"
            else:
                raise ValueError(
                    "Either pr_number or both base_commit and head_commit must be provided"
                )

            headers = self._get_headers(content_type="text/plain")

            response = requests.get(patch_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Handle rate limiting
            self._handle_rate_limit(response)

            return response.text

        return self._retry_wrapper(_get_patch)()

    def get_file_content(self, repo: str, commit: str, file_path: str) -> str:
        """
        Get the content of a file at a specific commit.

        Args:
            repo: Repository in format 'owner/repo'
            commit: Commit SHA to fetch the file from
            file_path: Path to the file in the repository

        Returns:
            File content as string

        Raises:
            requests.exceptions.RequestException: If the request fails
        """

        def _get_file_content():
            encoded_path = urllib.parse.quote(file_path, safe="")
            content_response = self.call_api(
                f"repos/{repo}/contents/{encoded_path}", params={"ref": commit}
            )
            content_data = content_response.json()

            # Decode base64 content
            if "content" in content_data and content_data.get("encoding") == "base64":
                import base64

                content = base64.b64decode(content_data["content"]).decode("utf-8")
                return content
            else:
                logger.warning(f"Unable to decode content for {file_path}")
                return ""

        return self._retry_wrapper(_get_file_content)()

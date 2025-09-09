import json
import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

try:
    import openai
except ImportError:
    raise ImportError(
        "OpenAI package not found. Please install it with: pip install openai"
    )

try:
    import anthropic
except ImportError:
    raise ImportError(
        "Anthropic package not found. Please install it with: pip install anthropic"
    )

try:
    import tiktoken
except ImportError:
    raise ImportError(
        "Tiktoken package not found. Please install it with: pip install tiktoken"
    )

DEFAULT_MAX_RETRIES = 10


class BaseModelClient(ABC):
    """Abstract base class for LLM model clients."""

    client: Any
    model: str
    model_provider: str
    model_kwargs: dict[str, Any]
    max_retries: int

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        self.model = model
        self.model_provider = model_provider
        self.model_kwargs = model_kwargs
        self.max_retries = max_retries

    @abstractmethod
    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using the LLM API.

        Args:
            messages: List of messages in OpenAI format [{"role": "user", "content": "..."}]

        Returns:
            The generated completion text
        """
        pass

    @abstractmethod
    def create_completion_with_structured_output(
        self, messages: list[dict[str, str]], json_schema: dict
    ) -> dict:
        """Create a completion with structured output using the LLM API.

        Args:
            messages: List of messages in OpenAI format [{"role": "user", "content": "..."}]
            json_schema: JSON Schema that defines the expected output structure

        Returns:
            The generated completion as a dictionary matching the schema
        """
        pass

    @abstractmethod
    def count_tokens_from_text(self, text: str) -> int:
        """Count the number of tokens in the text."""
        pass

    @abstractmethod
    def count_tokens_from_messages(self, messages: list[dict[str, str]]) -> int:
        """Count the number of tokens in the messages."""
        pass


class OpenAIClient(BaseModelClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        # Initialize the OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key, max_retries=self.max_retries)

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.model_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error creating OpenAI completion: {e}")
            raise e

    def create_completion_with_structured_output(
        self, messages: list[dict[str, str]], json_schema: dict, strict: bool = False
    ) -> dict:
        """Create a completion with structured output using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": json_schema.get("name", "structured_response"),
                        "strict": strict,
                        "schema": json_schema,
                    },
                },
                **self.model_kwargs,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error creating OpenAI structured completion: {e}")
            raise e

    def count_tokens_from_text(self, text: str) -> int:
        """Count the number of tokens in the text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fall back to o200k_base encoding if model not found
            encoding = tiktoken.get_encoding("o200k_base")

        return len(encoding.encode(text))

    def count_tokens_from_messages(self, messages: list[dict[str, str]]) -> int:
        """Count the number of tokens in the messages using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fall back to o200k_base encoding if model not found
            encoding = tiktoken.get_encoding("o200k_base")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


class DeepSeekClient(OpenAIClient):
    """DeepSeek API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        self.client = openai.OpenAI.copy(
            self.client, base_url="https://api.deepseek.com/v1"
        )


class QwenClient(OpenAIClient):
    """Qwen API client."""

    def __init__(self, model: str, model_provider: str, **model_kwargs: Any):
        # Handle enable_thinking
        if "enable_thinking" in model_kwargs:
            enable_thinking = model_kwargs.pop("enable_thinking")
            model_kwargs["extra_body"] = {"enable_thinking": enable_thinking}
        else:
            model_kwargs["extra_body"] = {"enable_thinking": False}

        super().__init__(model, model_provider, **model_kwargs)

        self.client = openai.OpenAI.copy(
            self.client, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using Qwen API with streaming support for enable_thinking."""
        # If enable_thinking is True, we need to use streaming
        if self.model_kwargs.get("extra_body", {}).get("enable_thinking", False):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **self.model_kwargs,
            )

            # Collect the streamed response
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content += chunk.choices[0].delta.content

            return content
        else:
            # Use parent implementation for non-thinking calls
            return super().create_completion(messages)


class AnthropicClient(BaseModelClient):
    """Anthropic API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        # Initialize the Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=self.max_retries)

    def _convert_to_anthropic_format(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], str | None]:
        """Convert messages from OpenAI format to Anthropic format.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            Tuple of (anthropic_messages, system_message)
        """
        system_message = None
        anthropic_messages = []

        # Extract system message if present
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]

        # Format remaining messages
        for msg in messages:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        return anthropic_messages, system_message

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using Anthropic API."""
        try:
            # Convert OpenAI format to Anthropic format
            anthropic_messages, system_message = self._convert_to_anthropic_format(
                messages
            )

            kwargs = self.model_kwargs.copy()
            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                max_tokens=kwargs.pop("max_tokens", 4096),
                **kwargs,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error creating Anthropic completion: {e}")
            raise e

    def create_completion_with_structured_output(
        self, messages: list[dict[str, str]], json_schema: dict
    ) -> dict:
        """Create a completion with structured output using Anthropic API."""
        try:
            # Convert OpenAI format to Anthropic format
            anthropic_messages, system_message = self._convert_to_anthropic_format(
                messages
            )

            kwargs = self.model_kwargs.copy()
            if system_message:
                kwargs["system"] = system_message

            # Create a tool definition from the JSON schema
            tool_name = json_schema.get("name", "record_output")
            tool = {
                "name": tool_name,
                "description": f"Record output using the schema: {json_schema.get('description', 'Structured output')}",
                "input_schema": json_schema,
            }

            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                max_tokens=kwargs.pop("max_tokens", 4096),
                tools=[tool],
                tool_choice={"type": "tool", "name": tool_name},
                **kwargs,
            )

            # Extract the tool use result
            for content in response.content:
                if content.type == "tool_use" and content.name == tool_name:
                    return content.input

            raise ValueError("No tool use found in response")
        except Exception as e:
            logger.error(f"Error creating Anthropic structured completion: {e}")
            raise e

    def count_tokens_from_text(self, text: str) -> int:
        """Count the number of tokens in the text using Anthropic's API."""
        try:
            # Wrap the text in a user message for token counting
            result = self.client.messages.count_tokens(
                model=self.model, messages=[{"role": "user", "content": text}]
            )
            return result.usage.input_tokens
        except Exception as e:
            logger.error(f"Error counting tokens with Anthropic API: {e}")
            raise e

    def count_tokens_from_messages(self, messages: list[dict[str, str]]) -> int:
        """Count the number of tokens in the messages using Anthropic's API."""
        try:
            # Convert OpenAI format to Anthropic format
            anthropic_messages, system_message = self._convert_to_anthropic_format(
                messages
            )

            # Count tokens with Anthropic API
            kwargs = {}
            if system_message:
                kwargs["system"] = system_message

            result = self.client.messages.count_tokens(
                model=self.model, messages=anthropic_messages, **kwargs
            )
            return result.usage.input_tokens
        except Exception as e:
            logger.error(f"Error counting tokens with Anthropic API: {e}")
            raise e

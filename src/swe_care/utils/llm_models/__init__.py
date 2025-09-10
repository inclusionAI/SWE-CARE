from typing import Any

from loguru import logger

from swe_care.utils.llm_models.clients import (
    AnthropicClient,
    BaseModelClient,
    DeepSeekClient,
    GeminiClient,
    MoonshotClient,
    OpenAIClient,
    QwenClient,
)

# Map of available LLM clients cited from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
LLM_CLIENT_MAP = {
    "openai": {
        "client_class": OpenAIClient,
        "models": [
            {"name": "gpt-4o", "max_input_tokens": 128000},
            {"name": "gpt-4o-mini", "max_input_tokens": 128000},
            {"name": "gpt-4.1", "max_input_tokens": 1047576},
            {"name": "gpt-4.5-preview", "max_input_tokens": 128000},
            {"name": "gpt-5", "max_input_tokens": 128000},
            {"name": "gpt-5-chat", "max_input_tokens": 128000},
            {"name": "o1", "max_input_tokens": 200000},
            {"name": "o1-mini", "max_input_tokens": 128000},
            {"name": "o3", "max_input_tokens": 200000},
            {"name": "o3-mini", "max_input_tokens": 200000},
        ],
    },
    "anthropic": {
        "client_class": AnthropicClient,
        "models": [
            {"name": "claude-opus-4-20250514", "max_input_tokens": 200000},
            {"name": "claude-sonnet-4-20250514", "max_input_tokens": 200000},
            {"name": "claude-3-7-sonnet-20250219", "max_input_tokens": 200000},
            {"name": "claude-3-5-sonnet-20241022", "max_input_tokens": 200000},
        ],
    },
    "deepseek": {
        "client_class": DeepSeekClient,
        "models": [
            {"name": "deepseek-chat", "max_input_tokens": 65536},
            {"name": "deepseek-reasoner", "max_input_tokens": 65536},
        ],
    },
    "qwen": {
        "client_class": QwenClient,
        "models": [
            {"name": "qwen3-32b", "max_input_tokens": 128000},
            {"name": "qwen3-30b-a3b", "max_input_tokens": 128000},
            {"name": "qwen3-235b-a22b", "max_input_tokens": 128000},
        ],
    },
    "moonshot": {
        "client_class": MoonshotClient,
        "models": [
            {"name": "kimi-k2-0711-preview", "max_input_tokens": 131072},
            {"name": "kimi-k2-0905-preview", "max_input_tokens": 131072},
        ],
    },
    "gemini": {
        "client_class": GeminiClient,
        "models": [
            {"name": "gemini-2.5-pro", "max_input_tokens": 1048576},
        ],
    },
}


def get_available_models_and_providers() -> tuple[list[str], list[str]]:
    """Get available models and providers from LLM_CLIENT_MAP."""
    available_providers = list(LLM_CLIENT_MAP.keys())
    available_models = []
    for provider_info in LLM_CLIENT_MAP.values():
        available_models.extend([model["name"] for model in provider_info["models"]])
    return available_providers, available_models


def init_llm_client(
    model: str, model_provider: str, **model_kwargs: Any
) -> BaseModelClient:
    """Initialize an LLM client.

    Args:
        model: Model name
        model_provider: Provider name (openai, anthropic)
        **model_kwargs: Additional model arguments

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If the model provider or model is not supported
    """
    if model_provider not in LLM_CLIENT_MAP:
        raise ValueError(
            f"Unsupported model provider: {model_provider}. "
            f"Supported providers: {list(LLM_CLIENT_MAP.keys())}"
        )

    provider_info = LLM_CLIENT_MAP[model_provider]

    _, model_list = get_available_models_and_providers()

    # if model not in model_list:
    #     logger.warning(
    #         f"Model {model} not in known models for {model_provider}. "
    #         f"Known models: {provider_info['models']}. Proceeding anyway..."
    #     )

    client_class = provider_info["client_class"]
    return client_class(model, model_provider, **model_kwargs)


def parse_model_args(model_args_str: str | None) -> dict[str, Any]:
    """Parse model arguments string into a dictionary.

    Args:
        model_args_str: Comma-separated string of key=value pairs

    Returns:
        Dictionary of parsed arguments

    Example:
        "top_p=0.95,temperature=0.70" -> {"top_p": 0.95, "temperature": 0.70}
    """
    if not model_args_str:
        return {}

    args = {}
    for pair in model_args_str.split(","):
        if "=" not in pair:
            logger.warning(f"Skipping invalid model argument: {pair}")
            continue

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Try int first
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                args[key] = int(value)
            # Try float
            elif "." in value and value.replace(".", "").replace("-", "").isdigit():
                args[key] = float(value)
            # Try boolean
            elif value.lower() in ("true", "false"):
                args[key] = value.lower() == "true"
            # Keep as string
            else:
                args[key] = value
        except ValueError:
            args[key] = value

    return args

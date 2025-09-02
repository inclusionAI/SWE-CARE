"""
Utility functions for loading and rendering YAML-based prompt templates.
"""

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

# Initialize Jinja2 environment
_template_dir = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(_template_dir))


def load_prompt(template_name: str, **context: Any) -> str | tuple[str, str]:
    """
    Load a YAML-based prompt template and render it with the provided context.

    Args:
        template_name: Name of the YAML template file (without extension) in the templates directory
        **context: Context variables to pass to the Jinja2 template

    Returns:
        - str: Rendered text if the template has only a 'text' key
        - tuple[str, str]: (system_prompt, user_prompt) if the template has 'system' and 'user' keys
        - str: Rendered user_prompt if the template has only a 'user' key

    Raises:
        FileNotFoundError: If the template file doesn't exist
        ValueError: If the template structure is invalid
    """
    template_path = _template_dir / f"{template_name}.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Load YAML content
    with open(template_path) as f:
        template_data = yaml.safe_load(f)

    if not isinstance(template_data, dict):
        raise ValueError(f"Invalid template structure in {template_name}.yaml")

    # Handle different template structures
    if "text" in template_data:
        # Simple text template
        template = _jinja_env.from_string(template_data["text"])
        return template.render(**context)

    elif "user" in template_data:
        # User prompt with optional system prompt
        user_template = _jinja_env.from_string(template_data["user"])
        user_prompt = user_template.render(**context)

        if "system" in template_data:
            # Both system and user prompts
            system_template = _jinja_env.from_string(template_data["system"])
            system_prompt = system_template.render(**context)
            return (system_prompt, user_prompt)
        else:
            # Only user prompt
            return user_prompt

    else:
        raise ValueError(
            f"Template {template_name}.yaml must contain either 'text' or 'user' key"
        )

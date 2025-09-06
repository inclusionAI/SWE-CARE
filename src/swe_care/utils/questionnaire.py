import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm

from swe_care.schema.dataset import (
    CodeReviewTaskInstance,
)
from swe_care.utils.estimate import InvalidResponseError
from swe_care.utils.llm_models.clients import BaseModelClient
from swe_care.utils.prompt_loader import load_prompt


def complete_questionnaire(
    client: BaseModelClient, instance: CodeReviewTaskInstance
) -> dict[str, Any]:
    """Complete the code review questionnaire using LLM with structured output.

    Args:
        client: The LLM client to use for completion
        instance: The code review task instance

    Returns:
        A dictionary containing the completed questionnaire responses
    """
    # Load the questionnaire context
    context = load_questionnaire(instance, context_only=True)

    # Load the questionnaire schema
    schema_path = (
        Path(__file__).parent.parent
        / "templates"
        / "questionnaire"
        / "questionnaire_schema.json"
    )
    with open(schema_path) as f:
        full_schema = json.load(f)

    # Extract section information
    sections = {
        "section1": "Section 1 — Problem Statement and Patch Alignment",
        "section2": "Section 2 — Review Scope and Comment Coverage",
        "section3": "Section 3 — Defects Identified in the Patch",
        "section4": "Section 4 — Difficulty and Review Effort",
        "section5": "Section 5 — Overall Patch Quality and Risk",
        "section6": "Section 6 — Dataset Suitability",
        "section7": "Section 7 — Confidence",
    }

    results = {}

    # Process each section separately with progress tracking
    for section_key, section_title in tqdm(
        sections.items(),
        desc=f"Processing questionnaire sections for {instance.instance_id}",
    ):
        # Create a schema for just this section
        section_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "name": f"{section_key}_response",
            "title": f"Response for {section_title}",
            "type": "object",
            "properties": {section_key: full_schema["properties"][section_key]},
            # OpenAI JSON schema response_format requires root objects to disallow
            # additional properties explicitly.
            "additionalProperties": False,
            "required": [section_key],
        }

        # Create retry wrapper for this section
        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(InvalidResponseError),
            reraise=True,
        )
        def complete_section():
            # Load prompts from YAML template
            system_prompt, user_prompt = load_prompt(
                "questionnaire/llm_respondent",
                section_title=section_title,
                context=context,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                response = client.create_completion_with_structured_output(
                    messages, section_schema
                )

                # Validate that we got the expected section key
                if section_key not in response:
                    raise InvalidResponseError(
                        f"Response missing expected section key: {section_key}"
                    )

                return response[section_key]

            except Exception as e:
                if isinstance(e, InvalidResponseError):
                    raise
                logger.warning(f"Error completing {section_key}: {e}")
                raise InvalidResponseError(f"LLM call failed: {e}")

        # Try to complete the section
        try:
            results[section_key] = complete_section()
        except InvalidResponseError:
            logger.error(
                f"Failed to complete {section_key} after 3 attempts, setting to None"
            )
            results[section_key] = None
        except Exception as e:
            logger.error(f"Unexpected error completing {section_key}: {e}")
            results[section_key] = None

    return results


def load_questionnaire(
    instance: CodeReviewTaskInstance, context_only: bool = False
) -> str:
    """
    Render the code review annotation questionnaire template with the provided instance.

    Args:
        instance: A CodeReviewTaskInstance dataclass object used as the Jinja2 context under key 'instance'.
        context_only: If True, only render the context without the schema questions.

    Returns:
        Rendered Markdown string for the questionnaire.

    Raises:
        FileNotFoundError: If the questionnaire template file doesn't exist.
    """

    if not isinstance(instance, CodeReviewTaskInstance):
        # Be explicit to help callers catch mismatches early
        raise TypeError("instance must be a CodeReviewTaskInstance")

    template_dir = Path(__file__).parent.parent / "templates" / "questionnaire"
    template_name = "questionnaire.md.j2"
    template_path = template_dir / "questionnaire.md.j2"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Only load and render schema if context_only is False
    schema_md = ""
    if not context_only:
        # Load questionnaire schema and render to Markdown content
        schema_path = template_dir / "questionnaire_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Questionnaire schema file not found: {schema_path}"
            )

        schema_obj: dict[str, Any]
        with open(schema_path) as f:
            schema_obj = json.load(f)

        schema_md = _render_questionnaire_schema(schema_obj, instance)

    jinja_env = Environment(loader=FileSystemLoader(template_dir))
    template = jinja_env.get_template(template_name)
    return template.render(instance=instance, schema_md=schema_md)


def _render_questionnaire_schema(
    schema: dict[str, Any], instance: CodeReviewTaskInstance
) -> str:
    """Render the questionnaire JSON schema into Markdown.

    This produces human-readable sections and questions based on the schema
    (titles, enums/options, and input formats), closely matching our static
    questionnaire layout.
    """
    sections = []

    def render_options(
        enum_vals: list[Any], options: list[dict[str, Any]] | None = None
    ) -> str:
        # If explicit options are provided, render "value: label" pairs (used for integer scales with descriptions)
        if options:
            option_map = {opt.get("value"): opt.get("label") for opt in options}
            lines = []
            for val in enum_vals:
                label = option_map.get(val, str(val))
                lines.append(f"- [ ] {val}: {label}")
            return "\n".join(lines)

        # Otherwise, render simple choices. For string enums, avoid duplicate "value: value" formatting
        if all(isinstance(v, str) for v in enum_vals):
            return "\n".join(f"- [ ] {v}" for v in enum_vals)
        # Fallback for numeric enums without options
        return "\n".join(f"- [ ] {v}" for v in enum_vals)

    properties: dict[str, Any] = schema.get("properties", {})
    # Maintain section order by sorting keys like section1..sectionN
    for section_key in sorted(properties.keys()):
        section = properties[section_key]
        section_title = section.get("title", section_key)
        section_type = section.get("type")
        section_lines = [f"### {section_title}"]

        if section_type == "object":
            qprops: dict[str, Any] = section.get("properties", {})
            for q_key, q_schema in qprops.items():
                q_title = q_schema.get("title", q_key)
                q_type = q_schema.get("type")

                # Heading per question
                section_lines.append("")
                # Derive question label like Q1.1, Q2.3 from keys such as q1_1, q2_3
                q_number = (
                    q_key[1:].replace("_", ".") if q_key.startswith("q") else q_key
                )
                section_lines.append(f"- **Q{q_number} — {q_title}**")

                # Optional question description (e.g., Q2.3 instruction text)
                q_desc = q_schema.get("description")
                if isinstance(q_desc, str) and q_desc.strip():
                    section_lines.append("")
                    section_lines.append(q_desc.strip())

                if q_type == "integer":
                    enum_vals = q_schema.get("enum")
                    options = q_schema.get("options")
                    if enum_vals:
                        section_lines.append(render_options(enum_vals, options))
                    else:
                        # Render min-max selector for bounded integers (e.g., confidence 1-5)
                        minimum = q_schema.get("minimum")
                        maximum = q_schema.get("maximum")
                        if (
                            isinstance(minimum, int)
                            and isinstance(maximum, int)
                            and minimum <= maximum
                        ):
                            choices = "  ".join(
                                f"[ ] {i}" for i in range(minimum, maximum + 1)
                            )
                            section_lines.append(f"\n  Select one: {choices}")

                elif q_type == "string":
                    section_lines.append("\n  _Your notes:_")

                elif q_type == "object":
                    subprops = q_schema.get("properties", {})
                    # Common composite question patterns (answer + reason/notes)
                    # Render option lists for any enum-bearing sub-field (answer, time_estimate, effort_level, risk_level, etc.)
                    for sub_key, sub_schema in subprops.items():
                        enum_vals = sub_schema.get("enum")
                        if enum_vals:
                            section_lines.append(
                                render_options(enum_vals, sub_schema.get("options"))
                            )
                    # Reason/rationale/notes fields
                    if "reason" in subprops:
                        section_lines.append("\n  - **Why?**")
                    if "rationale" in subprops:
                        section_lines.append("\n  - **Rationale:**")
                    if "notes" in subprops:
                        section_lines.append("\n  - **Notes:**")
                    # Other explanatory string fields (e.g., explanation)
                    for sub_key, sub_schema in subprops.items():
                        if sub_key in {"reason", "rationale", "notes", "answer"}:
                            continue
                        if sub_schema.get("type") == "string":
                            label = (
                                sub_schema.get("title")
                                or sub_key.replace("_", " ").capitalize()
                            )
                            if not str(label).endswith(":"):
                                label = f"{label}:"
                            section_lines.append(f"\n  - {label}")

                elif q_type == "array":
                    items = q_schema.get("items", {})
                    items_type = items.get("type")
                    if items_type == "object":
                        # Render a standard table for object arrays (e.g., q2_3)
                        # Attempt to derive columns from required/properties
                        props = items.get("properties", {})
                        cols = list(props.keys())
                        if cols:
                            # Enhance headers for known fields with enum hints
                            def header_for(col_key: str) -> str:
                                base = col_key.capitalize().replace("_", " ")
                                if col_key in ("category", "verdict"):
                                    enum_vals = props.get(col_key, {}).get("enum")
                                    if enum_vals:
                                        joined = "/".join(str(v) for v in enum_vals)
                                        return f"{base} ({joined})"
                                return base

                            header = " | ".join(header_for(c) for c in cols)
                            sep = " | ".join(["---"] * len(cols))
                            section_lines.append("")
                            section_lines.append(f"| {header} |")
                            section_lines.append(f"| {sep} |")
                            # Provide empty rows; if this is section2.q2_3, align rows with number of comments
                            rows = 3
                            if section_key == "section2" and q_key == "q2_3":
                                try:
                                    rows = max(
                                        1,
                                        len(
                                            getattr(
                                                instance,
                                                "reference_review_comments",
                                                [],
                                            )
                                        ),
                                    )
                                except Exception:
                                    rows = 3
                            for row_idx in range(rows):
                                cells = [" "] * len(cols)
                                # Prefill index for q2_3 with 1..N
                                if section_key == "section2" and q_key == "q2_3":
                                    try:
                                        index_pos = cols.index("index")
                                        cells[index_pos] = str(row_idx + 1)
                                    except ValueError:
                                        pass
                                section_lines.append("| " + " | ".join(cells) + " |")
                    else:
                        # Simple list placeholder
                        section_lines.append("")
                        section_lines.append("  - [ ] Item 1:")
                        section_lines.append("  - [ ] Item 2:")
                        section_lines.append("  - [ ] Item 3:")

        elif section_type == "array":
            items_schema = section.get("items", {})
            if items_schema.get("type") == "object":
                # Render a detailed block with item properties (e.g., defects block)
                item_props: dict[str, Any] = items_schema.get("properties", {})
                required_fields: list[str] = items_schema.get("required", []) or []

                # Humanize field labels
                def label_for(field: str, schema_def: dict[str, Any]) -> str:
                    if field == "severity":
                        minimum = schema_def.get("minimum")
                        maximum = schema_def.get("maximum")
                        if minimum is not None and maximum is not None:
                            # Use ASCII hyphen to match legacy formatting (e.g., 1-5)
                            return f"Severity ({minimum}-{maximum}):"
                        return "Severity:"
                    if field == "files_locations":
                        return "Files/Locations:"
                    if field == "short_description":
                        return "Short description:"
                    if field == "suggested_fix":
                        return (
                            "Suggested fix (optional):"
                            if field not in required_fields
                            else "Suggested fix:"
                        )
                    return f"{field.replace('_', ' ').capitalize()}:"

                section_lines.append("")
                section_lines.append(
                    "Repeat the following block for each defect you identify."
                )
                section_lines.append("")
                section_lines.append("```text")
                section_lines.append("Defect N")
                # Render fields in a stable order
                for field_name in item_props.keys():
                    schema_def = item_props[field_name]
                    label_text = label_for(field_name, schema_def)
                    enum_vals = schema_def.get("enum")
                    if isinstance(enum_vals, list) and enum_vals:
                        # Inline checkbox options, space-separated to match legacy style
                        options_inline = "  ".join(f"[ ] {str(v)}" for v in enum_vals)
                        section_lines.append(f"- {label_text} {options_inline}")
                    else:
                        section_lines.append(f"- {label_text}")
                section_lines.append("```")
            else:
                # Fallback simple list placeholder
                section_lines.append("")
                section_lines.append("Repeat the following block for each item:")
                section_lines.append("")
                section_lines.append("```text")
                section_lines.append("Item N")
                section_lines.append("- Value:")
                section_lines.append("```")

        # Separator between sections
        sections.append("\n".join(section_lines))

    return "\n\n---\n\n".join(sections)

"""
Utilities for extracting structured data from chart images via VLMs.

Main entrypoint: extract_data_from_chart
"""

import base64
import hashlib
import json
import logging
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from anthropic import Anthropic
from openai import OpenAI

CHART_EXTRACTION_PROMPT = """Here is an image of a chart. 
Please extract the numerical data it represents and return it in TSV (tab-separated values) format with appropriate headers. 
Copy the headers exactly as they are in the image. 
IMPORTANT: For the TSV, use tab (\t) as the separator.
Remember: The sole output should be the TSV table surrounded by ```tsv ```. Nothing else.
"""

CHART_EXTRACTION_PROMPT_WB = """Here is an image of a chart. 
Please extract the numerical data it represents and return it in TSV (tab-separated values) format with appropriate headers. 
For time-series charts, extract datapoints for **all years without any gaps**, even if a year is **not explicitly shown on the x axis**.
For series column headers, **use the country name**.
When a value is missing, **use "nan"**.
Copy the headers exactly as they are in the image where applicable. 
IMPORTANT: For the TSV, use tab (\t) as the separator.
Remember: The sole output should be the TSV table surrounded by ```tsv ```. Nothing else.
"""

CHART_STRUCTURE_PROMPT = """Here is an image of a chart.

Your task is to recover ONLY the table *structure* (schema + row keys), not the numeric values.

Return a TSV skeleton where:
- The first row is the header row (column names). Do NOT use "_" in the header row.
- The first column is the x-axis field (e.g., Year).
- Each additional column is one series (use the legend/labels exactly as written).
- Include one row for every x-axis tick/category shown.
- Do NOT leave any gaps in x-axis values: if the x-axis is a numeric sequence (e.g., years), output every consecutive value from the minimum to the maximum shown (even if some ticks are not labeled).
- Put "_" for every data cell (all non-header, non-x-axis cells).
- Preserve the exact spelling, capitalization, punctuation, and ordering of headers as shown in the chart.
- Use tab characters (\\t) as separators (write real tabs, not spaces).

Output format rules:
- Output ONLY the TSV skeleton, surrounded by ```tsv
``` and nothing else.
"""


def _get_structured_extraction_prompt(structure: str) -> str:
    """Generate structured extraction prompt with TSV skeleton."""
    header = structure.splitlines()[0].strip()  # no "\n" needed

    return f"""Here is an image of a chart.
Fill in the missing values (marked as "_") in the TSV skeleton below using the data from the chart.

Rules:
- Do NOT add, remove, reorder, or rename any rows or columns.
- Keep the headers EXACTLY as given: {header}
- Replace each "_" with the correct numeric value from the chart.
- If a value is not present / cannot be determined from the chart, replace "_" with "nan".
- Use tab (\\t) as the separator.

TSV skeleton to complete (copy exactly; only replace "_" / "nan" values):
{structure}

Remember: The sole output should be the completed TSV table surrounded by ```tsv ```. Nothing else.
"""


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Cache directory for persistent storage across sessions
CACHE_DIR = Path(".cache/extract_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Separate cache directory for structure extraction
STRUCTURE_CACHE_DIR = Path(".cache/structure")
STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Session state: tracks which cached result index to return next for each cache key
_session_cache_indices: Dict[str, int] = {}


def encode_image_base64(image_path: Path) -> str:
    """Base64-encode image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_type(image_path: Path) -> str:
    """Get the image type."""
    image_type = image_path.suffix.lstrip(".").lower()
    if image_type == "jpg":
        image_type = "jpeg"
    return image_type


def get_model_client(model_name: str) -> Any:
    """Return a client object for the given model provider."""
    if model_name.startswith("qwen/") or model_name.startswith("bytedance-seed/"):
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    elif model_name.startswith("meta-llama/"):
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    elif model_name.startswith("gpt"):
        return OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    elif model_name.startswith("gemini"):
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    elif model_name.startswith("claude"):
        return Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _get_cache_key(
    image_path: Path,
    model: str,
    temperature: Optional[float],
    prompt: str,
    structure: Optional[str],
) -> str:
    """Generate a cache key from function parameters."""
    # Normalize image path to absolute path for consistency
    abs_image_path = str(image_path.resolve())
    # TinyChart uses a different prompt
    if model == "tinychart":
        prompt = "tinychart"
    # Use a repo-relative path (from the top-level data/ directory) so that
    # cache keys are stable across machines and checkout locations.
    data_marker = os.sep + "data" + os.sep
    if data_marker in abs_image_path:
        abs_image_path = "data" + os.sep + abs_image_path.split(data_marker, 1)[1]
    # Create a hash of all parameters
    key_string = f"{abs_image_path}|{model}|{temperature}|{prompt}{f'|{structure}' if structure else ''}"
    return hashlib.sha256(key_string.encode()).hexdigest()


def _get_structure_cache_key(
    image_path: Path,
    model: str,
) -> str:
    """Generate a cache key from function parameters."""
    # Normalize image path to absolute path for consistency
    abs_image_path = str(image_path.resolve())
    return hashlib.sha256(f"{abs_image_path}|{model}".encode()).hexdigest()


def _load_cache(cache_key: str, cache_dir: Path = CACHE_DIR) -> List[Dict[str, Any]]:
    """Load cached results from disk."""
    cache_file = cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading cache file {cache_file}: {e}")
            return []
    return []


def _save_cache(
    cache_key: str, results: List[Dict[str, Any]], cache_dir: Path = CACHE_DIR
) -> None:
    """Save cached results to disk."""
    cache_file = cache_dir / f"{cache_key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
    except IOError as e:
        logger.warning(f"Error saving cache file {cache_file}: {e}")


def _get_cached_result_or_generate(
    cache_key: str,
    generate_fn: Callable[[], Dict[str, Any]],
    cache_dir: Path = CACHE_DIR,
) -> Dict[str, Any]:
    """Get next cached result or generate a new one if cache is exhausted.

    Args:
        cache_key: The cache key for this set of parameters.
        generate_fn: Function to call to generate a new result.
        cache_dir: Directory to use for caching.

    Returns:
        A result dictionary.
    """
    # Load cached results
    cached_results = _load_cache(cache_key, cache_dir)

    # Get the current index for this cache key in this session
    current_index = _session_cache_indices.get(cache_key, 0)

    # If we have a cached result at this index, return it
    if current_index < len(cached_results):
        result = cached_results[current_index]
        # Increment index for next call
        _session_cache_indices[cache_key] = current_index + 1
        return result

    # Cache exhausted, generate new result
    result = generate_fn()

    # Append to cache and save
    cached_results.append(result)
    _save_cache(cache_key, cached_results, cache_dir)

    # Increment index for next call
    _session_cache_indices[cache_key] = current_index + 1

    return result


def _get_cached_structure_result_or_generate(
    cache_key: str,
    generate_fn: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Get cached structure result or generate a new one if not cached.

    Args:
        cache_key: The cache key for this set of parameters.
        generate_fn: Function to call to generate a new result.

    Returns:
        A structure dictionary.
    """
    # Load cached result
    cache_file = STRUCTURE_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                # Handle both old format (list) and new format (dict)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    return cached_data[0]
                elif isinstance(cached_data, dict):
                    return cached_data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading structure cache file {cache_file}: {e}")

    # Not cached, generate new result
    result = generate_fn()

    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2)
    except IOError as e:
        logger.warning(f"Error saving structure cache file {cache_file}: {e}")

    return result


def conforms_to_structure(tsv_data: str, structure: str) -> bool:
    """Check if the TSV data conforms to the structure:
    - The number of rows is the same as the number of rows in the structure.
    - The number of columns is the same as the number of columns in the structure.
    - The headers are the same as the headers in the structure.
    - The row keys are the same as the row keys in the structure.
    """
    structure_lines = [line for line in structure.strip().split("\n") if line.strip()]
    tsv_lines = [line for line in tsv_data.strip().split("\n") if line.strip()]
    if not structure_lines or not tsv_lines:
        return False

    # Check headers
    structure_header = structure_lines[0].rstrip()
    tsv_header = tsv_lines[0].rstrip()
    if structure_header != tsv_header:
        return False

    # Check number of rows (including header)
    if len(structure_lines) != len(tsv_lines):
        return False

    # Check number of columns for each row and that keys match
    for row_idx in range(1, len(structure_lines)):
        struct_row = structure_lines[row_idx].split("\t")
        tsv_row = tsv_lines[row_idx].split("\t")
        if len(struct_row) != len(tsv_row):
            return False
        # Row key (first column) must match
        if struct_row[0].strip() != tsv_row[0].strip():
            return False

    return True


def extract_data_from_chart(
    image_path: Path,
    model: str,
    temperature: Optional[float] = None,
    prompt: str = CHART_EXTRACTION_PROMPT,
    structure: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract data from a chart image."""
    # Use structured extraction prompt if structure is provided
    if structure is not None:
        prompt = _get_structured_extraction_prompt(structure)

    cache_key = _get_cache_key(image_path, model, temperature, prompt, structure)
    # print("Cache key:", cache_key)

    def _generate_result() -> Dict[str, Any]:
        """Generate a new result by calling the LLM."""
        client = get_model_client(model)
        base64_image = encode_image_base64(image_path)
        image_type = get_image_type(image_path)
        data_url = f"data:image/{image_type};base64,{base64_image}"

        if model.startswith("claude"):
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{image_type}",
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                **({"temperature": temperature} if temperature is not None else {}),
            )
            try:
                content = response.content[0].text
                if "```tsv" in content:
                    csv_data = content.split("```tsv")[1].split("```")[0].strip()
                else:
                    csv_data = content.split("```")[1].strip()
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
            except Exception as e:
                logger.error(
                    f"Unexpected error during data extraction: {e}", exc_info=True
                )
                csv_data = None
                input_tokens = 0
                output_tokens = 0
        elif model.startswith("gemini"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                                "detail": "media_resolution_high",
                            },
                        ],
                    }
                ],
                reasoning_effort="low",
            )
            try:
                content = response.choices[0].message.content
                if "```tsv" in content:
                    csv_data = content.split("```tsv")[1].split("```")[0].strip()
                else:
                    csv_data = content.split("```")[1].strip()
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.total_tokens - input_tokens
            except JSONDecodeError:
                logger.error(f"Error decoding response (JSONDecodeError): {response}")
                csv_data = None
                input_tokens = 0
                output_tokens = 0
            except KeyError:
                logger.error(f"Error decoding response (KeyError): {response.json()}")
                csv_data = None
                input_tokens = 0
                output_tokens = 0
            except Exception as e:
                logger.error(
                    f"Unexpected error during data extraction: {e}", exc_info=True
                )
                csv_data = None
                input_tokens = 0
                output_tokens = 0
        else:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
                temperature=temperature,
                reasoning=({"effort": "none"} if model.startswith("gpt") else None),
            )

            try:
                content = response.output_text
                if "```tsv" in content:
                    csv_data = content.split("```tsv")[1].split("```")[0].strip()
                else:
                    csv_data = content.split("```")[1].strip()
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
            except JSONDecodeError:
                logger.error(f"Error decoding response (JSONDecodeError): {response}")
                csv_data = None
                input_tokens = 0
                output_tokens = 0
            except KeyError:
                logger.error(f"Error decoding response (KeyError): {response.json()}")
                csv_data = None
                input_tokens = 0
                output_tokens = 0
            except Exception as e:
                logger.error(
                    f"Unexpected error during data extraction: {e}", exc_info=True
                )
                csv_data = None
                input_tokens = 0
                output_tokens = 0

        return {
            "csv_data": csv_data,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    if structure is not None:
        cached_results = _load_cache(cache_key, CACHE_DIR)

        current_index = _session_cache_indices.get(cache_key, 0)

        # If we have a cached result at this index, return it (no validation/filtering)
        if current_index < len(cached_results):
            result = cached_results[current_index]
            _session_cache_indices[cache_key] = current_index + 1
            return result

        # Cache exhausted -> generate with validation + retry, and ONLY cache if valid
        max_retries = 2
        retry_count = 0

        while retry_count <= max_retries:
            result = _generate_result()

            if result.get("csv_data") and conforms_to_structure(
                result["csv_data"], structure
            ):
                cached_results.append(result)
                _save_cache(cache_key, cached_results, CACHE_DIR)
                _session_cache_indices[cache_key] = current_index + 1
                return result

            if retry_count < max_retries:
                logger.warning(
                    f"Extraction result does not conform to structure (attempt {retry_count + 1}/{max_retries + 1}). "
                    f"Retrying extraction for {image_path}.\n"
                )
            retry_count += 1

        logger.warning(
            f"Extraction result still does not conform to structure after {max_retries} retries "
            f"for {image_path}. Returning result anyway (NOT cached)."
        )
        return result

    # No structure provided, use normal caching
    return _get_cached_result_or_generate(cache_key, _generate_result)


def extract_structure_from_chart(
    image_path: Path,
    model: str,
) -> Dict[str, Any]:
    """Extract structure from a chart image."""
    cache_key = _get_structure_cache_key(image_path, model)

    def _generate_structure_result() -> Dict[str, Any]:
        """Generate a new result by calling the LLM."""
        client = get_model_client(model)
        base64_image = encode_image_base64(image_path)
        image_type = get_image_type(image_path)
        data_url = f"data:image/{image_type};base64,{base64_image}"

        if model.startswith("claude"):
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{image_type}",
                                    "data": base64_image,
                                },
                            },
                            {"type": "text", "text": CHART_STRUCTURE_PROMPT},
                        ],
                    }
                ],
            )
            content = response.content[0].text
        elif model.startswith("gemini"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CHART_STRUCTURE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                                "detail": "media_resolution_high",
                            },
                        ],
                    }
                ],
                reasoning_effort="low",
            )
            content = response.choices[0].message.content
        else:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": CHART_STRUCTURE_PROMPT},
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
                reasoning={"effort": "medium"},
            )
            content = response.output_text
        if "```tsv" in content:
            structure = content.split("```tsv")[1].split("```")[0].strip()
        else:
            structure = content.split("```")[1].strip()

        lines = [line for line in structure.split("\n") if line.strip() != ""]
        if lines:
            # Replace underscores with space in the first row (header)
            header = lines[0].replace("_", " ")
            # Replace underscores with space in the first column for all rows
            modified_lines = [header]
            for row in lines[1:]:
                cols = row.split("\t")
                if cols:
                    cols[0] = cols[0].replace("_", "")
                modified_lines.append("\t".join(cols))
            structure = "\n".join(modified_lines)

        return {"structure": structure}

    # Use caching: return cached result if available, otherwise generate new one
    return _get_cached_structure_result_or_generate(
        cache_key, _generate_structure_result
    )

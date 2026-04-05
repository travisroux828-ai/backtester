"""
Shared utilities for AI modules.
"""

from __future__ import annotations

import json
import re


def parse_response(text: str) -> dict:
    """Extract and parse JSON from Claude's response text."""
    # Try parsing the whole response first
    stripped = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", stripped)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Extract JSON from a code fence within surrounding text
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Extract the first top-level JSON object from the text
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[brace_start:i + 1])

    raise json.JSONDecodeError("No JSON object found in response", text, 0)

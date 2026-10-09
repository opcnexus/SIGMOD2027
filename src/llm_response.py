#!/usr/bin/env python3
"""Shared LLM response classification: separate "output truncation" from "network / empty content /
bad response" as decidable failure modes.

P0-1 fix (formal review Sec. 10): all previous ECB variants never checked `finish_reason`,
so when the output was truncated by `max_tokens` the JSON was incomplete -> parse failure -> the whole
conversation degraded -> the metrics were recorded as 0,
counting engineering failures as model capability. This module provides the single implementation
shared by the pilot / full version and the regression tests.
"""
MAX_TOKENS_CAP = 8192

FAILURE_MODES = ("network", "output_truncated", "empty_content", "bad_response")


def classify_response(body):
    """Extract (content, finish_reason, failure_mode) from an OpenAI-compatible response body.

    Returns:
    content       -- the model output text on success; on failure possibly None or truncated partial text
    finish_reason -- the raw finish_reason (stop / length etc. on success)
    failure_mode  -- None means success, otherwise one of FAILURE_MODES
    """
    try:
        ch = body["choices"][0]
    except Exception:
        return None, None, "bad_response"
    fr = ch.get("finish_reason")
    content = (ch.get("message") or {}).get("content")
    if fr == "length":
        # the output was truncated by max_tokens: the JSON is necessarily incomplete and must be
        # distinguished from a network failure
        return content, fr, "output_truncated"
    if content is None or content == "":
        return None, fr, "empty_content"
    return content, fr, None

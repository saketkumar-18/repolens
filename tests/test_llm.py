"""Tests for LLM client helpers (no network)."""

from __future__ import annotations

from repolens.llm import extract_json


def test_extract_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_in_fence():
    text = 'Here you go:\n```json\n{"a": [1, 2]}\n```\nDone.'
    assert extract_json(text) == {"a": [1, 2]}


def test_extract_json_with_prose_around():
    text = 'Sure! {"comments": [], "summary": {"verdict": "approve"}} hope that helps'
    out = extract_json(text)
    assert out is not None and out["summary"]["verdict"] == "approve"


def test_extract_json_nested_braces_in_strings():
    text = '{"msg": "use { and } carefully", "n": 1}'
    assert extract_json(text) == {"msg": "use { and } carefully", "n": 1}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("no json here at all") is None


def test_extract_json_ignores_arrays():
    assert extract_json("[1, 2, 3]") is None

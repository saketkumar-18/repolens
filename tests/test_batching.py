"""Tests for batched LLM review of large PRs and LLM client guardrails."""

from __future__ import annotations

import pytest

from repolens.llm import LLMClient, LLMError
from repolens.reviewer import ReviewConfig, Reviewer


class CountingStubLLM:
    """Records how many batches it was called with."""

    def __init__(self):
        self.calls: list[int] = []  # number of files per call

    def chat_json(self, messages, **kwargs):
        user = messages[-1]["content"]
        n_files = user.count("### MODIFIED:") + user.count("### ADDED:") + user.count("### RENAMED:")
        self.calls.append(n_files)
        return {
            "summary": {"verdict": "comment", "overview": "batch ok", "strengths": ["s"]},
            "comments": [],
        }, "stub", 100


def _diff_touching(n_files: int) -> str:
    parts = []
    for i in range(n_files):
        parts.append(
            f"""diff --git a/f{i}.py b/f{i}.py
--- a/f{i}.py
+++ b/f{i}.py
@@ -1 +1,2 @@
 x = 1
+y{i} = {i}
"""
        )
    return "\n".join(parts)


def test_batching_splits_files(fixture_repo):
    stub = CountingStubLLM()
    cfg = ReviewConfig()
    cfg.batch_size = 2
    cfg.max_files = 10
    reviewer = Reviewer(config=cfg, llm=stub)  # type: ignore[arg-type]
    result = reviewer.review_local(str(fixture_repo), diff_text=_diff_touching(5))
    assert result.files_reviewed == 5
    assert stub.calls == [2, 2, 1]
    assert result.tokens_used == 300  # 3 batches x 100


def test_batch_failure_does_not_stop_others(fixture_repo):
    class FailFirstLLM:
        def __init__(self):
            self.n = 0

        def chat_json(self, messages, **kwargs):
            self.n += 1
            if self.n == 1:
                raise LLMError("transient boom")
            return {"summary": {"verdict": "approve", "overview": "ok", "strengths": []}, "comments": []}, "stub", 50

    cfg = ReviewConfig()
    cfg.batch_size = 1
    reviewer = Reviewer(config=cfg, llm=FailFirstLLM())  # type: ignore[arg-type]
    result = reviewer.review_local(str(fixture_repo), diff_text=_diff_touching(2))
    assert result.files_reviewed == 2
    assert result.model == "stub"  # second batch succeeded


def test_request_changes_verdict_wins_across_batches(fixture_repo):
    class MixedVerdictLLM:
        def __init__(self):
            self.n = 0

        def chat_json(self, messages, **kwargs):
            self.n += 1
            verdict = "approve" if self.n == 1 else "request_changes"
            return {"summary": {"verdict": verdict, "overview": "x", "strengths": []}, "comments": []}, "stub", 10

    cfg = ReviewConfig()
    cfg.batch_size = 1
    reviewer = Reviewer(config=cfg, llm=MixedVerdictLLM())  # type: ignore[arg-type]
    result = reviewer.review_local(str(fixture_repo), diff_text=_diff_touching(2))
    assert result.summary.verdict == "request_changes"


def test_no_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("REPOLENS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(api_key="")
    with pytest.raises(LLMError, match="no API key"):
        client.chat([{"role": "user", "content": "hi"}])

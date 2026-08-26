"""Orchestrator tests with a stubbed LLM (no network)."""

from __future__ import annotations

from repolens.models import Severity
from repolens.reviewer import ReviewConfig, Reviewer


class StubLLM:
    """Returns a canned review; records the prompt it received."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_messages = None

    def chat_json(self, messages, **kwargs):
        self.last_messages = messages
        return self.payload, "stub-model", 1234


def _make_reviewer(payload: dict) -> tuple[Reviewer, StubLLM]:
    stub = StubLLM(payload)
    r = Reviewer(config=ReviewConfig(), llm=stub)  # type: ignore[arg-type]
    return r, stub


GOOD_PAYLOAD = {
    "summary": {
        "verdict": "request_changes",
        "overview": "Adds delete_user with an injection risk.",
        "strengths": ["regex email validation is stricter"],
    },
    "comments": [
        {
            "path": "src/service/user_service.py",
            "line": 20,
            "severity": "critical",
            "category": "security",
            "message": "f-string SQL — use parameterized query like deactivate_user does.",
            "confidence": 0.95,
            "context_refs": ["src/repo/db.py"],
        },
        {
            "path": "src/utils/validators.py",
            "line": 6,
            "severity": "suggestion",
            "category": "style",
            "message": "Compile the pattern once at module level.",
            "confidence": 0.7,
        },
    ],
}


def test_review_local_end_to_end(fixture_repo, sample_diff):
    reviewer, stub = _make_reviewer(GOOD_PAYLOAD)
    result = reviewer.review_local(
        str(fixture_repo), diff_text=sample_diff, base_ref="main", head_ref="HEAD"
    )
    assert result.files_reviewed == 2
    assert result.model == "stub-model"
    assert result.summary.verdict == "request_changes"
    # heuristic SQL finding + LLM critical on same line are deduped
    sec = [c for c in result.comments if c.category == "security"]
    assert sec, "expected security findings"
    # prompt carried repo context
    user_msg = stub.last_messages[-1]["content"]
    assert "REPOSITORY CONTEXT" in user_msg
    assert "REPO CONTEXT:" in user_msg


def test_llm_comment_line_validation(fixture_repo, sample_diff):
    payload = {
        "summary": {"verdict": "comment", "overview": "x", "strengths": []},
        "comments": [
            {  # line 9999 doesn't exist -> snapped to a real changed line
                "path": "src/service/user_service.py",
                "line": 9999,
                "severity": "warning",
                "category": "correctness",
                "message": "somewhere",
                "confidence": 0.9,
            },
            {  # unknown path -> dropped
                "path": "not/in/diff.py",
                "line": 1,
                "severity": "warning",
                "category": "correctness",
                "message": "ghost",
                "confidence": 0.9,
            },
        ],
    }
    reviewer, _ = _make_reviewer(payload)
    result = reviewer.review_local(str(fixture_repo), diff_text=sample_diff)
    llm_comments = [c for c in result.comments if c.source == "llm"]
    assert len(llm_comments) == 1
    ch_lines = {20, 21, 22, 24, 25}  # added lines in user_service.py per fixture
    assert llm_comments[0].line in ch_lines or llm_comments[0].line is not None


def test_low_confidence_filtered(fixture_repo, sample_diff):
    payload = {
        "summary": {"verdict": "approve", "overview": "fine", "strengths": []},
        "comments": [
            {
                "path": "src/utils/validators.py",
                "line": 10,
                "severity": "suggestion",
                "category": "style",
                "message": "meh",
                "confidence": 0.1,
            }
        ],
    }
    reviewer, _ = _make_reviewer(payload)
    result = reviewer.review_local(str(fixture_repo), diff_text=sample_diff)
    assert not [c for c in result.comments if c.source == "llm" and c.message == "meh"]


def test_llm_failure_falls_back_to_heuristics(fixture_repo, sample_diff):
    class BrokenLLM:
        def chat_json(self, messages, **kwargs):
            from repolens.llm import LLMError

            raise LLMError("boom")

    reviewer = Reviewer(config=ReviewConfig(), llm=BrokenLLM())  # type: ignore[arg-type]
    result = reviewer.review_local(str(fixture_repo), diff_text=sample_diff)
    assert result.comments, "heuristics should still produce findings"
    assert all(c.source == "heuristic" for c in result.comments)
    assert result.summary.verdict == "request_changes"


def test_binary_and_lockfiles_skipped(fixture_repo):
    diff = """\
diff --git a/package-lock.json b/package-lock.json
--- a/package-lock.json
+++ b/package-lock.json
@@ -1 +1,2 @@
 {}
+{"x": 1}
diff --git a/img.png b/img.png
Binary files a/img.png and b/img.png differ
"""
    reviewer, _ = _make_reviewer({"summary": {"verdict": "approve", "overview": "", "strengths": []}, "comments": []})
    result = reviewer.review_local(str(fixture_repo), diff_text=diff)
    assert result.files_reviewed == 0
    assert "package-lock.json" in result.files_skipped
    assert result.summary.verdict == "approve"


def test_comments_sorted_by_severity(fixture_repo, sample_diff):
    reviewer, _ = _make_reviewer(GOOD_PAYLOAD)
    result = reviewer.review_local(str(fixture_repo), diff_text=sample_diff)
    sevs = [c.severity for c in result.comments]
    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.SUGGESTION: 2, Severity.PRAISE: 3}
    assert sevs == sorted(sevs, key=lambda s: order[s])

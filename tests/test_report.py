"""Tests for report rendering and the poster payload builder."""

from __future__ import annotations

from repolens.models import ReviewComment, ReviewResult, ReviewSummary, Severity
from repolens.poster import post_review
from repolens.report import render_comment_body, render_full_report, render_summary_comment


def _result() -> ReviewResult:
    return ReviewResult(
        repo="acme/widget",
        pr_number=7,
        base_ref="main",
        head_ref="feat",
        head_sha="abc1234567890",
        files_reviewed=2,
        comments=[
            ReviewComment(
                path="a.py",
                line=10,
                severity=Severity.CRITICAL,
                category="security",
                message="SQL injection here.",
                suggestion="cur.execute(q, (param,))",
                context_refs=["db.py"],
            ),
            ReviewComment(
                path="b.py",
                line=None,
                severity=Severity.SUGGESTION,
                category="style",
                message="File-level note.",
            ),
        ],
        summary=ReviewSummary(verdict="request_changes", overview="Needs fixes.", strengths=["nice tests"]),
        model="stub",
        context_files_used=["db.py"],
    )


def test_comment_body_has_suggestion_block():
    body = render_comment_body(_result().comments[0])
    assert "```suggestion" in body
    assert "`db.py`" in body


def test_summary_comment_structure():
    md = render_summary_comment(_result())
    assert "RepoLens" in md
    assert "REQUEST CHANGES" in md
    assert "a.py:10" in md
    assert "nice tests" in md


def test_full_report():
    md = render_full_report(_result())
    assert md.startswith("# RepoLens Review")
    assert "PR #7" in md
    assert "SQL injection here." in md


def test_post_dry_run_payload():
    out = post_review(_result(), "acme/widget", 7, dry_run=True)
    assert out["dry_run"] is True
    assert out["event"] == "REQUEST_CHANGES"
    assert out["commit_id"] == "abc1234567890"
    # file-level comment (line=None) excluded from inline list
    assert len(out["comments"]) == 1
    assert out["comments"][0]["path"] == "a.py"
    assert out["comments"][0]["side"] == "RIGHT"

"""Pydantic models shared across RepoLens."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DiffLine(BaseModel):
    """One line inside a hunk."""

    kind: Literal["add", "del", "ctx"]
    text: str
    new_lineno: int | None = None  # line number in the new file (None for deletions)
    old_lineno: int | None = None  # line number in the old file (None for additions)


class Hunk(BaseModel):
    """A @@ hunk with parsed line ranges."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str = ""  # the text after the second @@ (function/class context)
    lines: list[DiffLine] = Field(default_factory=list)

    @property
    def added_lines(self) -> list[DiffLine]:
        return [ln for ln in self.lines if ln.kind == "add"]

    @property
    def deleted_lines(self) -> list[DiffLine]:
        return [ln for ln in self.lines if ln.kind == "del"]


class FileChange(BaseModel):
    """One changed file in a PR."""

    path: str  # new path (or old path for deletions)
    old_path: str | None = None  # set when renamed
    status: Literal["added", "modified", "deleted", "renamed"] = "modified"
    hunks: list[Hunk] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False
    new_content: str | None = None  # full file at head, when available

    @property
    def changed_line_numbers(self) -> set[int]:
        """New-file line numbers touched by this change (for comment validation)."""
        nums: set[int] = set()
        for h in self.hunks:
            for ln in h.lines:
                if ln.kind == "add" and ln.new_lineno is not None:
                    nums.add(ln.new_lineno)
        return nums

    def diff_text(self) -> str:
        """Re-render this file's unified diff."""
        out: list[str] = []
        header_old = self.old_path or self.path
        out.append(f"--- a/{header_old}")
        out.append(f"+++ b/{self.path}")
        for h in self.hunks:
            out.append(
                f"@@ -{h.old_start},{h.old_count} +{h.new_start},{h.new_count} @@ {h.section}".rstrip()
            )
            for ln in h.lines:
                prefix = {"add": "+", "del": "-", "ctx": " "}[ln.kind]
                out.append(f"{prefix}{ln.text}")
        return "\n".join(out)


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"
    PRAISE = "praise"


class ReviewComment(BaseModel):
    """A single line-level review comment."""

    path: str
    line: int | None = None  # new-file line number; None => file-level comment
    severity: Severity = Severity.SUGGESTION
    category: str = "general"  # correctness | security | performance | style | testing | docs | general
    message: str
    suggestion: str | None = None  # optional replacement code snippet
    source: Literal["llm", "heuristic"] = "llm"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    context_refs: list[str] = Field(default_factory=list)  # repo files that informed this comment


class ReviewSummary(BaseModel):
    """PR-level verdict."""

    verdict: Literal["approve", "request_changes", "comment"] = "comment"
    overview: str = ""
    strengths: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Full output of one review run."""

    repo: str = ""
    pr_number: int | None = None
    base_ref: str = ""
    head_ref: str = ""
    head_sha: str = ""
    files_reviewed: int = 0
    files_skipped: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    summary: ReviewSummary = Field(default_factory=ReviewSummary)
    model: str = ""
    context_files_used: list[str] = Field(default_factory=list)
    tokens_used: int = 0

    @property
    def blocking_count(self) -> int:
        return sum(1 for c in self.comments if c.severity in (Severity.CRITICAL, Severity.WARNING))


class RepoFile(BaseModel):
    """A file indexed from the repository."""

    path: str
    language: str
    content: str
    symbols: list[str] = Field(default_factory=list)
    size: int = 0

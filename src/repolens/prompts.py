"""Prompt construction for the repo-aware reviewer."""

from __future__ import annotations

import json

from .models import FileChange

SYSTEM_PROMPT = """\
You are RepoLens, an expert senior software engineer performing a code review.

You review a pull request diff WITH FULL REPOSITORY CONTEXT: besides the diff
you are given the complete contents of changed files and the most relevant
chunks from other files in the repository (retrieved by symbol/identifier
relevance). Use that repo context to catch issues that are invisible from the
diff alone: broken callers, duplicated logic, contract violations, missing
updates to related code, inconsistent conventions.

Rules:
1. Comment ONLY on lines that the diff adds or modifies (kind "+"). Reference
   exact line numbers from the provided line-numbered diff.
2. Every comment must cite the concrete code it refers to. No vague advice.
3. Severities:
   - "critical": bugs, security holes, data loss, crashes — must fix before merge.
   - "warning": likely bugs, race conditions, error-handling gaps, bad practices.
   - "suggestion": improvements, readability, minor refactors, style.
   - "praise": genuinely well-done pieces (use sparingly, max 2).
4. Prefer few, high-signal comments. Do NOT pad. Nitpicks about formatting are
   unwelcome unless the project enforces them.
5. When repo context shows the change breaks or duplicates something elsewhere,
   say so and name the file in context_refs.
6. If the change is clean, say so — an empty comments list with an approving
   summary is a valid review.
7. Never invent file paths or line numbers that are not in the provided material.

Respond with STRICT JSON only (no markdown fences, no commentary outside JSON):
{
  "summary": {
    "verdict": "approve" | "request_changes" | "comment",
    "overview": "2-4 sentence assessment of the change as a whole",
    "strengths": ["..."]
  },
  "comments": [
    {
      "path": "src/foo.py",
      "line": 42,
      "severity": "critical|warning|suggestion|praise",
      "category": "correctness|security|performance|style|testing|docs|general",
      "message": "what is wrong and why, referencing repo context where relevant",
      "suggestion": "optional replacement code snippet",
      "confidence": 0.0-1.0,
      "context_refs": ["other/file.py"]
    }
  ]
}
"""


def _numbered_diff(change: FileChange) -> str:
    """Render a file's diff with new-side line numbers for grounding."""
    out: list[str] = []
    for h in change.hunks:
        out.append(f"@@ -{h.old_start},{h.old_count} +{h.new_start},{h.new_count} @@ {h.section}".rstrip())
        for ln in h.lines:
            if ln.kind == "add":
                out.append(f"{ln.new_lineno:>6} + {ln.text}")
            elif ln.kind == "del":
                out.append(f"       - {ln.text}")
            else:
                out.append(f"{ln.new_lineno:>6}   {ln.text}")
    return "\n".join(out)


def build_review_prompt(
    changes: list[FileChange],
    context_pack: str,
    pr_title: str = "",
    pr_body: str = "",
    base_ref: str = "main",
    head_ref: str = "feature",
) -> list[dict]:
    """Build the chat messages for one review call."""
    diff_sections: list[str] = []
    for ch in changes:
        status = ch.status.upper()
        old = f" (renamed from {ch.old_path})" if ch.old_path and ch.status == "renamed" else ""
        diff_sections.append(f"### {status}: {ch.path}{old}\n{_numbered_diff(ch)}")

    pr_meta = ""
    if pr_title:
        pr_meta = f"PR title: {pr_title}\n"
    if pr_body:
        pr_meta += f"PR description:\n{pr_body[:2000]}\n"

    user = f"""Pull request: {base_ref} <- {head_ref}
{pr_meta}
Files changed: {len(changes)}

=================== DIFF (line numbers = new file) ===================
{chr(10).join(diff_sections)}

=================== REPOSITORY CONTEXT ===================
{context_pack if context_pack else "(no additional repo context available)"}

=================== TASK ===================
Review the diff using the repository context. Return the JSON review object now."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def comment_schema_hint() -> str:
    """JSON schema hint (kept for docs/tests)."""
    return json.dumps(
        {
            "summary": {"verdict": "str", "overview": "str", "strengths": ["str"]},
            "comments": [
                {
                    "path": "str",
                    "line": "int|null",
                    "severity": "critical|warning|suggestion|praise",
                    "category": "str",
                    "message": "str",
                    "suggestion": "str|null",
                    "confidence": "float",
                    "context_refs": ["str"],
                }
            ],
        }
    )

"""Deterministic static heuristics.

A cheap, LLM-free first pass that catches the classic footguns with zero
hallucination risk. Findings are merged with LLM comments (de-duplicated by
path+line) and clearly tagged source="heuristic".
"""

from __future__ import annotations

import re

from .models import FileChange, ReviewComment, Severity

_SECRET_PATTERNS = [
    (re.compile(r"""(?:api[_-]?key|apikey|secret|passwd|password|token|auth)""", re.I), None),
]
_SECRET_ASSIGN = re.compile(
    r"""(?i)\b(api[_-]?key|apikey|secret(?:_key)?|password|passwd|pwd|access[_-]?token|auth[_-]?token|private[_-]?key|client[_-]?secret)\b
    \s*[:=]\s*['"][A-Za-z0-9_\-./+=]{12,}['"]""",
    re.X,
)
_KNOWN_KEYS = re.compile(
    r"""(?x)
    \b(?:
      sk-[A-Za-z0-9]{20,}                       # openai
      | sk-or-v1-[A-Za-z0-9\-]{20,}             # openrouter
      | ghp_[A-Za-z0-9]{30,}                    # github PAT
      | gho_[A-Za-z0-9]{30,}
      | github_pat_[A-Za-z0-9_]{30,}
      | xox[baprs]-[A-Za-z0-9\-]{10,}           # slack
      | AKIA[0-9A-Z]{16}                        # aws
      | AIza[0-9A-Za-z_\-]{35}                  # google api
      | -----BEGIN\ [A-Z\ ]*PRIVATE\ KEY-----
    )""",
)
_TODO_MARK = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
_DEBUG_PRINT_PY = re.compile(r"^\s*(print|pprint)\s*\(")
_DEBUG_PRINT_JS = re.compile(r"\bconsole\.(log|debug)\s*\(")
_DEBUGGER = re.compile(r"^\s*(debugger|breakpoint\(\)|import\s+ipdb|pdb\.set_trace\(\))")
_EVAL = re.compile(r"\beval\s*\(")
_EXEC = re.compile(r"\bexec\s*\(")
_SQL_FSTRING = re.compile(r"""(?i)(execute|cursor\.execute|query)\s*\(\s*f['"]""")
_MERGE_MARKERS = re.compile(r"^(<{7}|>{7}|={7})( |$)")
_EMPTY_EXCEPT = re.compile(r"^\s*except\s*(\w+\s*)?:\s*(pass|\.\.\.)?\s*$")
_NOQA_SKIP = re.compile(r"#\s*nosec|# noqa", re.I)


def run_heuristics(changes: list[FileChange]) -> list[ReviewComment]:
    """Scan added lines for deterministic issues."""
    out: list[ReviewComment] = []
    for ch in changes:
        if ch.is_binary:
            continue
        lang = ch.path.rsplit(".", 1)[-1].lower() if "." in ch.path else ""
        for h in ch.hunks:
            for ln in h.added_lines:
                text = ln.text
                if ln.new_lineno is None:
                    continue
                line_no = ln.new_lineno

                if _KNOWN_KEYS.search(text) or _SECRET_ASSIGN.search(text):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.CRITICAL,
                            category="security",
                            message="Hardcoded secret/credential detected. Move it to an environment "
                            "variable or secret manager and rotate the exposed value.",
                            source="heuristic",
                            confidence=0.95,
                        )
                    )
                if _MERGE_MARKERS.match(text):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.CRITICAL,
                            category="correctness",
                            message="Unresolved merge-conflict markers committed to the file.",
                            source="heuristic",
                            confidence=0.99,
                        )
                    )
                if _SQL_FSTRING.search(text):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.CRITICAL,
                            category="security",
                            message="SQL query built with an f-string — SQL injection risk. "
                            "Use parameterized queries instead.",
                            source="heuristic",
                            confidence=0.85,
                        )
                    )
                if lang == "py" and (_EVAL.search(text) or _EXEC.search(text)) and not text.strip().startswith("#"):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.WARNING,
                            category="security",
                            message="Use of eval()/exec() on dynamic input is dangerous. "
                            "Prefer ast.literal_eval or an explicit parser.",
                            source="heuristic",
                            confidence=0.7,
                        )
                    )
                if _DEBUGGER.search(text):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.WARNING,
                            category="correctness",
                            message="Debugger/breakpoint statement left in code — remove before merge.",
                            source="heuristic",
                            confidence=0.95,
                        )
                    )
                if (lang == "py" and _DEBUG_PRINT_PY.match(text)) or (
                    lang in ("js", "ts", "jsx", "tsx", "mjs", "cjs") and _DEBUG_PRINT_JS.search(text)
                ):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.SUGGESTION,
                            category="style",
                            message="Debug print/console.log left in the change. "
                            "Remove it or replace with proper logging.",
                            source="heuristic",
                            confidence=0.6,
                        )
                    )
                if _EMPTY_EXCEPT.match(text):
                    out.append(
                        ReviewComment(
                            path=ch.path,
                            line=line_no,
                            severity=Severity.WARNING,
                            category="correctness",
                            message="Bare `except: pass` swallows every error silently. "
                            "Catch specific exceptions and log or re-raise.",
                            source="heuristic",
                            confidence=0.8,
                        )
                    )
    return out

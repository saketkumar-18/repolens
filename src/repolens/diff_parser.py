"""Unified-diff parser with precise old/new line-number tracking.

Accepts either a full multi-file unified diff (as produced by
``git diff`` / the GitHub ``.diff`` endpoint) or a single-file patch.
Produces :class:`repolens.models.FileChange` objects whose line numbers
match the *new* side of the file — the coordinate system GitHub review
comments use.
"""

from __future__ import annotations

import re

from .models import DiffLine, FileChange, Hunk

_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_HUNK_HDR = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_OLD_FILE = re.compile(r"^--- (?:a/)?(.+)$")
_NEW_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_RENAME_FROM = re.compile(r"^rename from (.+)$")
_RENAME_TO = re.compile(r"^rename to (.+)$")
_SIMILARITY = re.compile(r"^similarity index \d+%$")


def parse_diff(text: str) -> list[FileChange]:
    """Parse a unified diff into structured FileChange objects."""
    files: list[FileChange] = []
    current: FileChange | None = None
    hunk: Hunk | None = None
    old_ln = new_ln = 0
    pending_old: str | None = None
    pending_new: str | None = None
    rename_from: str | None = None
    rename_to: str | None = None
    in_header = True  # per-file header region (before first @@)

    def flush() -> None:
        nonlocal current, hunk, pending_old, pending_new, rename_from, rename_to, in_header
        if current is not None:
            if pending_old and pending_new and pending_old != pending_new:
                current.old_path = pending_old
                if current.status == "modified":
                    current.status = "renamed"
            if rename_from and rename_to and rename_from != rename_to:
                current.old_path = rename_from
                current.path = rename_to
                current.status = "renamed"
            current.additions = sum(len(h.added_lines) for h in current.hunks)
            current.deletions = sum(len(h.deleted_lines) for h in current.hunks)
            files.append(current)
        current = None
        hunk = None
        pending_old = None
        pending_new = None
        rename_from = None
        rename_to = None
        in_header = True

    for raw in text.splitlines():
        m = _DIFF_GIT.match(raw)
        if m:
            flush()
            current = FileChange(path=m.group(2))
            if m.group(1) != m.group(2):
                current.old_path = m.group(1)
            continue

        if current is None:
            # Tolerate diffs without the `diff --git` preamble.
            mo = _OLD_FILE.match(raw)
            if mo and mo.group(1) != "/dev/null":
                if files and files[-1].hunks:
                    flush()
                current = FileChange(path=mo.group(1))
                pending_old = mo.group(1)
                continue
            continue

        if in_header:
            if raw.startswith("Binary files"):
                current.is_binary = True
                continue
            if raw == "new file mode" or raw.startswith("new file mode"):
                current.status = "added"
                continue
            if raw.startswith("deleted file mode"):
                current.status = "deleted"
                continue
            mr = _RENAME_FROM.match(raw)
            if mr:
                rename_from = mr.group(1)
                continue
            mr = _RENAME_TO.match(raw)
            if mr:
                rename_to = mr.group(1)
                continue
            if _SIMILARITY.match(raw) or raw.startswith("index ") or raw.startswith("old mode") or raw.startswith("new mode"):
                continue
            mo = _OLD_FILE.match(raw)
            if mo:
                pending_old = None if mo.group(1) == "/dev/null" else mo.group(1)
                continue
            mn = _NEW_FILE.match(raw)
            if mn:
                pending_new = None if mn.group(1) == "/dev/null" else mn.group(1)
                if pending_new:
                    current.path = pending_new
                if pending_old is None and pending_new and current.status == "modified":
                    current.status = "added"
                continue

        mh = _HUNK_HDR.match(raw)
        if mh:
            in_header = False
            hunk = Hunk(
                old_start=int(mh.group(1)),
                old_count=int(mh.group(2) or 1),
                new_start=int(mh.group(3)),
                new_count=int(mh.group(4) or 1),
                section=mh.group(5).strip(),
            )
            current.hunks.append(hunk)
            old_ln, new_ln = hunk.old_start, hunk.new_start
            continue

        if hunk is None:
            continue

        if raw.startswith("+") and not raw.startswith("+++"):
            hunk.lines.append(DiffLine(kind="add", text=raw[1:], new_lineno=new_ln))
            new_ln += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            hunk.lines.append(DiffLine(kind="del", text=raw[1:], old_lineno=old_ln))
            old_ln += 1
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        else:
            content = raw[1:] if raw.startswith(" ") else raw
            hunk.lines.append(DiffLine(kind="ctx", text=content, new_lineno=new_ln, old_lineno=old_ln))
            old_ln += 1
            new_ln += 1

    flush()
    return files


def validate_comment_line(change: FileChange, line: int | None) -> int | None:
    """Clamp/validate an LLM-proposed comment line to a commentable position.

    GitHub only accepts comment lines that exist in the diff. If the exact
    line is not in the diff we snap to the nearest added line; if the file
    has no added lines (pure deletion) we return None (file-level comment).
    """
    if line is None:
        return None
    changed = sorted(change.changed_line_numbers)
    if not changed:
        return None
    if line in change.changed_line_numbers:
        return line
    # snap to nearest changed line
    return min(changed, key=lambda n: abs(n - line))

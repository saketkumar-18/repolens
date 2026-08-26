"""Tests for the unified-diff parser."""

from __future__ import annotations

from repolens.diff_parser import parse_diff, validate_comment_line

MULTI_DIFF = """\
diff --git a/a.py b/a.py
index 111..222 100644
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@ def foo():
 line1
-old
+new
+added
 line3
diff --git a/old.txt b/new.txt
similarity index 80%
rename from old.txt
rename to new.txt
index 333..444 100644
--- a/old.txt
+++ b/new.txt
@@ -1 +1 @@
-x
+y
diff --git a/gone.py b/gone.py
deleted file mode 100644
index 555..0000000
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-a
-b
diff --git a/fresh.py b/fresh.py
new file mode 100644
index 0000000..666
--- /dev/null
+++ b/fresh.py
@@ -0,0 +1,2 @@
+hello
+world
"""


def test_parse_multi_file_diff():
    files = parse_diff(MULTI_DIFF)
    assert [f.path for f in files] == ["a.py", "new.txt", "gone.py", "fresh.py"]


def test_line_numbers_track_new_side():
    files = parse_diff(MULTI_DIFF)
    a = files[0]
    adds = {ln.new_lineno: ln.text for h in a.hunks for ln in h.added_lines}
    assert adds == {2: "new", 3: "added"}
    dels = [ln.text for h in a.hunks for ln in h.deleted_lines]
    assert dels == ["old"]
    assert a.hunks[0].section == "def foo():"


def test_statuses():
    files = parse_diff(MULTI_DIFF)
    by_path = {f.path: f for f in files}
    assert by_path["a.py"].status == "modified"
    assert by_path["new.txt"].status == "renamed"
    assert by_path["new.txt"].old_path == "old.txt"
    assert by_path["gone.py"].status == "deleted"
    assert by_path["fresh.py"].status == "added"


def test_addition_deletion_counts():
    files = parse_diff(MULTI_DIFF)
    a = files[0]
    assert a.additions == 2
    assert a.deletions == 1


def test_diff_roundtrip_contains_markers():
    files = parse_diff(MULTI_DIFF)
    text = files[0].diff_text()
    assert "--- a/a.py" in text
    assert "+++ b/a.py" in text
    assert "+added" in text


def test_validate_comment_line_exact():
    files = parse_diff(MULTI_DIFF)
    a = files[0]
    assert validate_comment_line(a, 3) == 3


def test_validate_comment_line_snaps():
    files = parse_diff(MULTI_DIFF)
    a = files[0]
    # line 100 not in diff -> snaps to nearest changed line (3)
    assert validate_comment_line(a, 100) == 3


def test_validate_deleted_file_returns_none():
    files = parse_diff(MULTI_DIFF)
    gone = files[2]
    assert validate_comment_line(gone, 1) is None


def test_no_newline_marker_ignored():
    diff = """\
diff --git a/f.py b/f.py
--- a/f.py
+++ b/f.py
@@ -1 +1 @@
-a
\\ No newline at end of file
+b
\\ No newline at end of file
"""
    files = parse_diff(diff)
    assert files[0].additions == 1
    assert files[0].deletions == 1

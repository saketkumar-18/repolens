"""Tests for the static heuristics pass."""

from __future__ import annotations

from repolens.diff_parser import parse_diff
from repolens.heuristics import run_heuristics
from repolens.models import Severity


def _comments_for(diff: str):
    return run_heuristics(parse_diff(diff))


def test_detects_hardcoded_secret():
    diff = """\
diff --git a/cfg.py b/cfg.py
--- a/cfg.py
+++ b/cfg.py
@@ -1 +1,2 @@
 x = 1
+API_KEY = "sk-or...cdef"
"""
    cs = _comments_for(diff)
    assert any(c.severity == Severity.CRITICAL and c.category == "security" for c in cs)


def test_detects_sql_fstring():
    diff = """\
diff --git a/db.py b/db.py
--- a/db.py
+++ b/db.py
@@ -1 +1,2 @@
 pass
+    cursor.execute(f"SELECT * FROM t WHERE id = {uid}")
"""
    cs = _comments_for(diff)
    assert any(c.category == "security" and "SQL" in c.message for c in cs)


def test_detects_debug_print():
    diff = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1,2 @@
 pass
+    print("debug here")
"""
    cs = _comments_for(diff)
    assert any(c.category == "style" and "print" in c.message for c in cs)


def test_detects_console_log_in_js():
    diff = """\
diff --git a/app.js b/app.js
--- a/app.js
+++ b/app.js
@@ -1 +1,2 @@
 const a = 1;
+console.log("debug");
"""
    cs = _comments_for(diff)
    assert any("console.log" in c.message for c in cs)


def test_detects_merge_conflict_markers():
    diff = """\
diff --git a/m.py b/m.py
--- a/m.py
+++ b/m.py
@@ -1 +1,3 @@
 pass
+<<<<<<< HEAD
+ours
"""
    cs = _comments_for(diff)
    assert any(c.severity == Severity.CRITICAL and "merge" in c.message.lower() for c in cs)


def test_detects_empty_except():
    diff = """\
diff --git a/e.py b/e.py
--- a/e.py
+++ b/e.py
@@ -1 +1,2 @@
 pass
+    except: pass
"""
    cs = _comments_for(diff)
    assert any("except" in c.message for c in cs)


def test_clean_diff_no_findings():
    diff = """\
diff --git a/ok.py b/ok.py
--- a/ok.py
+++ b/ok.py
@@ -1 +1,2 @@
 x = 1
+y = compute_total(x)
"""
    assert _comments_for(diff) == []


def test_heuristic_lines_match_diff(sample_diff):
    """The planted SQL injection + print in the fixture diff are caught."""
    cs = _comments_for(sample_diff)
    paths = {(c.path, c.category) for c in cs}
    assert ("src/service/user_service.py", "security") in paths
    assert ("src/service/user_service.py", "style") in paths

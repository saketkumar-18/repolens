"""Tests for the reference-graph retrieval signal (caller boost + call graph)."""

from __future__ import annotations

from repolens.diff_parser import parse_diff
from repolens.repo_index import iter_repo_files
from repolens.retrieval import RepoContextRetriever


def test_callers_of_changed_symbols(fixture_repo, sample_diff):
    """user_service.py defines UserService; routes.py + tests reference it."""
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    retr = RepoContextRetriever(repo_files)
    # only the user_service.py change defines symbols with callers
    callers = retr.callers_of_changed_symbols(changes[:1])
    assert "UserService" in callers
    assert "src/api/routes.py" in callers["UserService"]
    assert "tests/test_user_service.py" in callers["UserService"]
    # the defining file itself is never its own caller
    assert "src/service/user_service.py" not in callers["UserService"]


def test_caller_files_get_boosted(fixture_repo, sample_diff):
    """Chunks in caller files outrank equally-relevant non-caller files."""
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    retr = RepoContextRetriever(repo_files)
    chunks, _ = retr.retrieve(changes[:1], top_k=10)
    caller_chunks = [c for c in chunks if c.get("is_caller")]
    assert caller_chunks, "expected at least one caller chunk in top-k"
    # every caller chunk must be flagged consistently with the reference graph
    callers = retr.callers_of_changed_symbols(changes[:1])
    caller_paths = {p for paths in callers.values() for p in paths}
    for c in caller_chunks:
        assert c["path"] in caller_paths


def test_context_pack_contains_call_graph_section(fixture_repo, sample_diff):
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    for ch in changes:
        p = fixture_repo / ch.path
        if p.exists():
            ch.new_content = p.read_text(encoding="utf-8")
    retr = RepoContextRetriever(repo_files, context_budget=30_000)
    pack, used = retr.assemble_context(changes[:1], top_k=8)
    assert "CALL GRAPH" in pack
    assert "UserService" in pack
    assert "src/api/routes.py" in pack
    # caller files count toward used context files
    assert any("routes" in u or "test_user_service" in u for u in used)


def test_no_callers_no_call_graph_section(fixture_repo):
    """A change to a file nobody references produces no CALL GRAPH block."""
    repo_files = iter_repo_files(fixture_repo)
    diff = """\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # demo repo
+more docs
"""
    changes = parse_diff(diff)
    retr = RepoContextRetriever(repo_files, context_budget=30_000)
    pack, _ = retr.assemble_context(changes, top_k=4)
    assert "CALL GRAPH" not in pack

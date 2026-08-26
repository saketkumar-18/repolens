"""Tests for repo indexing + symbol-aware retrieval (the capstone)."""

from __future__ import annotations

from repolens.diff_parser import parse_diff
from repolens.repo_index import chunk_file, detect_language, extract_symbols, iter_repo_files
from repolens.retrieval import BM25Index, RepoContextRetriever, tokenize


def test_detect_language():
    assert detect_language("src/a.py") == "python"
    assert detect_language("app.tsx") == "typescript"
    assert detect_language("Dockerfile") == "docker"
    assert detect_language("x.unknown") == "text"


def test_extract_python_symbols():
    code = "class Foo:\n    def bar(self):\n        pass\n\nasync def baz():\n    pass\n"
    syms = extract_symbols(code, "python")
    assert "Foo" in syms and "bar" in syms and "baz" in syms


def test_extract_js_symbols():
    code = "export function doThing() {}\nconst helper = () => 1;\nclass Widget {}\n"
    syms = extract_symbols(code, "javascript")
    assert "doThing" in syms and "helper" in syms and "Widget" in syms


def test_tokenize_splits_camel_and_snake():
    toks = tokenize("getUserById some_price")
    assert "getuserbyid" in toks
    assert "user" in toks
    assert "some" in toks and "price" in toks


def test_tokenize_drops_stopwords():
    toks = tokenize("return self value data")
    assert toks == []


def test_chunk_file_overlap():
    content = "\n".join(f"line{i}" for i in range(100))
    chunks = chunk_file("f.py", content, "python")
    assert len(chunks) >= 3
    assert chunks[0]["start"] == 1
    # overlap: second chunk starts before first ends
    assert chunks[1]["start"] < chunks[0]["end"]


def test_iter_repo_files_skips_dirs(fixture_repo):
    (fixture_repo / "node_modules").mkdir()
    (fixture_repo / "node_modules" / "junk.js").write_text("x", encoding="utf-8")
    (fixture_repo / "binary.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    files = iter_repo_files(fixture_repo)
    paths = {f.path for f in files}
    assert "src/service/user_service.py" in paths
    assert "node_modules/junk.js" not in paths
    assert "binary.bin" not in paths


def test_bm25_ranks_relevant_higher():
    idx = BM25Index()
    idx.add({"id": "a"}, "def validate_email(email): return True")
    idx.add({"id": "b"}, "database connection pooling configuration")
    idx.build()
    results = idx.query(tokenize("validate_email"), top_k=2)
    assert results[0][1]["id"] == "a"


def test_retriever_excludes_changed_files(fixture_repo, sample_diff):
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    retr = RepoContextRetriever(repo_files)
    chunks, terms = retr.retrieve(changes, top_k=10)
    changed_paths = {c.path for c in changes}
    for c in chunks:
        assert c["path"] not in changed_paths
    # query should contain identifiers from the diff
    assert any("delete_user" in t or "username" in t for t in terms)


def test_context_pack_contains_changed_file_and_context(fixture_repo, sample_diff):
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    for ch in changes:
        p = fixture_repo / ch.path
        if p.exists():
            ch.new_content = p.read_text(encoding="utf-8")
    retr = RepoContextRetriever(repo_files, context_budget=20_000)
    pack, used = retr.assemble_context(changes, top_k=8)
    assert "FILE (changed): src/service/user_service.py" in pack
    assert "REPO CONTEXT:" in pack
    assert used  # at least one cross-file chunk used


def test_context_budget_respected(fixture_repo, sample_diff):
    repo_files = iter_repo_files(fixture_repo)
    changes = parse_diff(sample_diff)
    retr = RepoContextRetriever(repo_files, context_budget=500)
    pack, _ = retr.assemble_context(changes, top_k=8)
    assert len(pack) <= 500 + 100  # small tolerance for truncation marker


def test_symbol_definitions_lookup(fixture_repo):
    repo_files = iter_repo_files(fixture_repo)
    retr = RepoContextRetriever(repo_files)
    assert "src/repo/db.py" in retr.symbol_definitions("get_connection")
    assert "src/utils/validators.py" in retr.symbol_definitions("validate_email")

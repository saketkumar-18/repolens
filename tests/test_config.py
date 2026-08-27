"""Tests for per-repo configuration (.repolens.toml)."""

from __future__ import annotations

from pathlib import Path

from repolens.config import apply_config, filter_paths, load_repo_config, path_matches
from repolens.reviewer import ReviewConfig


def test_load_missing_returns_empty(tmp_path: Path):
    assert load_repo_config(tmp_path) == {}


def test_load_valid_toml(tmp_path: Path):
    (tmp_path / ".repolens.toml").write_text(
        """
context_budget = 80000
max_files = 15
min_confidence = 0.5
include_heuristics = false
model = "openai/gpt-4o"

[focus]
paths = ["src/core/**"]
ignore = ["**/generated/**"]
""",
        encoding="utf-8",
    )
    cfg = load_repo_config(tmp_path)
    assert cfg["context_budget"] == 80000
    assert cfg["focus"]["ignore"] == ["**/generated/**"]


def test_load_malformed_toml_returns_empty(tmp_path: Path):
    (tmp_path / ".repolens.toml").write_text("this is [ not toml", encoding="utf-8")
    assert load_repo_config(tmp_path) == {}


def test_apply_config_sets_fields():
    cfg = ReviewConfig()
    apply_config(
        cfg,
        {
            "context_budget": 1234,
            "max_files": 3,
            "min_confidence": 0.9,
            "include_heuristics": False,
            "model": "m",
            "batch_size": 2,
        },
    )
    assert cfg.context_budget == 1234
    assert cfg.max_files == 3
    assert cfg.min_confidence == 0.9
    assert cfg.include_heuristics is False
    assert cfg.model == "m"
    assert cfg.batch_size == 2


def test_apply_config_ignores_wrong_types():
    cfg = ReviewConfig()
    before = cfg.context_budget
    apply_config(cfg, {"context_budget": "not-an-int", "model": ""})
    assert cfg.context_budget == before
    assert cfg.model is None


def test_path_matches_glob():
    assert path_matches("src/core/x.py", ["src/core/**"])
    assert not path_matches("src/other/x.py", ["src/core/**"])
    assert path_matches("a/b/generated/c.py", ["**/generated/**"])


def test_filter_paths_include_and_ignore():
    paths = ["src/core/a.py", "src/core/generated/gen.py", "src/web/b.py"]
    out = filter_paths(paths, {"paths": ["src/core/**"], "ignore": ["**/generated/**"]})
    assert out == ["src/core/a.py"]


def test_review_local_picks_up_repo_config(fixture_repo, sample_diff):
    """A .repolens.toml in the reviewed repo changes reviewer behavior."""
    from repolens.reviewer import Reviewer

    (fixture_repo / ".repolens.toml").write_text(
        """
include_heuristics = false

[focus]
ignore = ["src/utils/**"]
""",
        encoding="utf-8",
    )

    class StubLLM:
        def chat_json(self, messages, **kwargs):
            return {"summary": {"verdict": "approve", "overview": "ok", "strengths": []}, "comments": []}, "stub", 1

    reviewer = Reviewer(config=ReviewConfig(), llm=StubLLM())  # type: ignore[arg-type]
    result = reviewer.review_local(str(fixture_repo), diff_text=sample_diff)
    # validators.py (src/utils/**) was ignored via repo config
    assert "src/utils/validators.py" in result.files_skipped
    assert result.files_reviewed == 1
    # heuristics disabled via repo config
    assert all(c.source != "heuristic" for c in result.comments)

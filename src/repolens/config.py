"""Per-repo configuration via `.repolens.toml` at the repo root.

Example:

    # .repolens.toml
    context_budget = 80000
    max_files = 15
    top_k_chunks = 32
    min_confidence = 0.5
    include_heuristics = true
    model = "openai/gpt-4o"

    [focus]
    paths = ["src/core/**"]        # only review these paths (glob)
    ignore = ["**/generated/**"]   # never review these paths

Values here override the built-in defaults but are themselves overridden by
explicit CLI flags / API request fields.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    tomllib = None  # type: ignore[assignment]

CONFIG_NAME = ".repolens.toml"

_BOOL_KEYS = ("include_heuristics",)
_INT_KEYS = ("context_budget", "max_files", "top_k_chunks", "batch_size")
_FLOAT_KEYS = ("min_confidence",)
_STR_KEYS = ("model",)


def load_repo_config(repo_path: str | Path) -> dict:
    """Load .repolens.toml from a repo root. Returns {} when absent/unreadable."""
    if tomllib is None:
        return {}
    cfg_path = Path(repo_path) / CONFIG_NAME
    if not cfg_path.is_file():
        return {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def apply_config(config, repo_cfg: dict) -> None:
    """Apply a loaded repo config dict onto a ReviewConfig in place."""
    for key in _INT_KEYS:
        if isinstance(repo_cfg.get(key), int):
            setattr(config, key, repo_cfg[key])
    for key in _FLOAT_KEYS:
        if isinstance(repo_cfg.get(key), (int, float)):
            setattr(config, key, float(repo_cfg[key]))
    for key in _BOOL_KEYS:
        if isinstance(repo_cfg.get(key), bool):
            setattr(config, key, repo_cfg[key])
    for key in _STR_KEYS:
        if isinstance(repo_cfg.get(key), str) and repo_cfg[key]:
            setattr(config, key, repo_cfg[key])


def path_matches(path: str, patterns: list[str]) -> bool:
    """Glob match supporting ** segments."""
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        # allow matching on any path suffix for patterns like "generated/**"
        if "**" in pat and fnmatch.fnmatch(path, pat.replace("**/", "**")):
            continue
    return False


def filter_paths(paths: list[str], focus: dict) -> list[str]:
    """Apply [focus] paths/ignore globs from repo config."""
    include = focus.get("paths") or []
    ignore = focus.get("ignore") or []
    out: list[str] = []
    for p in paths:
        if ignore and path_matches(p, ignore):
            continue
        if include and not path_matches(p, include):
            continue
        out.append(p)
    return out

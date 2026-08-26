"""Repository indexing: language detection, symbol extraction, chunking.

Pure-stdlib implementation (no tree-sitter dependency) so the agent runs
anywhere. Symbol extraction is regex-based per language family and covers
the common definition forms; it is deliberately tolerant — a missed symbol
only slightly degrades retrieval recall, it never breaks the pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import RepoFile

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

EXT_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".rst": "markdown",
    ".dockerfile": "docker",
    ".tf": "terraform",
    ".lua": "lua",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
}

CODE_LANGS = {
    "python", "javascript", "typescript", "java", "kotlin", "go", "rust",
    "ruby", "php", "c", "cpp", "csharp", "swift", "scala", "shell", "sql",
    "vue", "svelte", "lua", "dart", "elixir",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", ".nuxt", "out", "target",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "vendor", "coverage", ".eggs", "site-packages", ".turbo", ".cache",
    ".repolens",
}

MAX_FILE_BYTES = 400_000  # skip files larger than this


def detect_language(path: str) -> str:
    name = Path(path).name.lower()
    if name == "dockerfile":
        return "docker"
    if name in ("makefile", "gnumakefile"):
        return "make"
    return EXT_LANG.get(Path(path).suffix.lower(), "text")


# ---------------------------------------------------------------------------
# Symbol extraction (regex per language family)
# ---------------------------------------------------------------------------

_PY_DEF = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)
_PY_CLASS = re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)
_JS_FUNC = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
)
_JS_CONST_FN = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
)
_JS_CLASS = re.compile(r"(?:^|\n)\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)")
_JAVA_DECL = re.compile(
    r"^\s*(?:public|private|protected|static|final|abstract|\s)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_$][\w$]*)",
    re.M,
)
_JAVA_METHOD = re.compile(
    r"^\s*(?:public|private|protected|static|final|abstract|synchronized|\s)+"
    r"[\w<>\[\],\s]+\s+([a-z][\w$]*)\s*\([^;{]*\)\s*\{",
    re.M,
)
_GO_FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)", re.M)
_GO_TYPE = re.compile(r"^type\s+([A-Za-z_]\w*)\s+(?:struct|interface)", re.M)
_RUST_FN = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([a-z_]\w*)", re.M)
_RUST_TYPE = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|type)\s+([A-Z]\w*)", re.M)
_RB_DEF = re.compile(r"^\s*(?:def\s+(?:self\.)?([A-Za-z_]\w*)|class\s+([A-Za-z_]\w*)|module\s+([A-Za-z_]\w*))", re.M)
_PHP_FUNC = re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*function\s+([A-Za-z_]\w*)", re.M)
_C_FUNC = re.compile(r"^[A-Za-z_][\w\s\*]*?\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", re.M)
_SQL_TABLE = re.compile(r"(?:create|CREATE)\s+(?:or\s+replace\s+)?(?:table|TABLE|view|VIEW)\s+(?:if\s+not\s+exists\s+)?[\"`']?([\w.]+)", re.I)


def extract_symbols(content: str, language: str) -> list[str]:
    """Extract defined symbol names from file content. Best-effort."""
    names: list[str] = []

    def add(matches: list[str]) -> None:
        for m in matches:
            if isinstance(m, tuple):
                m = next((g for g in m if g), "")
            if m:
                names.append(m)

    if language == "python":
        add(_PY_DEF.findall(content))
        add(_PY_CLASS.findall(content))
    elif language in ("javascript", "typescript", "vue", "svelte"):
        add(_JS_FUNC.findall(content))
        add(_JS_CONST_FN.findall(content))
        add(_JS_CLASS.findall(content))
    elif language in ("java", "kotlin", "csharp", "scala"):
        add(_JAVA_DECL.findall(content))
        add(_JAVA_METHOD.findall(content))
    elif language == "go":
        add(_GO_FUNC.findall(content))
        add(_GO_TYPE.findall(content))
    elif language == "rust":
        add(_RUST_FN.findall(content))
        add(_RUST_TYPE.findall(content))
    elif language == "ruby":
        for tup in _RB_DEF.findall(content):
            add([tup])
    elif language == "php":
        add(_PHP_FUNC.findall(content))
    elif language in ("c", "cpp"):
        add(_C_FUNC.findall(content))
    elif language == "sql":
        add(_SQL_TABLE.findall(content))

    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:200]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_CHUNK_LINES = 40
_CHUNK_OVERLAP = 8


def chunk_file(path: str, content: str, language: str) -> list[dict]:
    """Split a file into overlapping line chunks for retrieval.

    Each chunk: {path, start, end, text, symbols}.
    """
    lines = content.splitlines()
    if not lines:
        return []
    chunks: list[dict] = []
    step = _CHUNK_LINES - _CHUNK_OVERLAP
    for start in range(0, len(lines), step):
        end = min(start + _CHUNK_LINES, len(lines))
        text = "\n".join(lines[start:end])
        chunks.append(
            {
                "path": path,
                "start": start + 1,
                "end": end,
                "text": text,
                "symbols": extract_symbols(text, language),
            }
        )
        if end >= len(lines):
            break
    return chunks


# ---------------------------------------------------------------------------
# Repo walking
# ---------------------------------------------------------------------------


def iter_repo_files(root: str | Path, include_globs: list[str] | None = None) -> list[RepoFile]:
    """Walk a repo directory and return indexable files (respects SKIP_DIRS)."""
    root = Path(root)
    out: list[RepoFile] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(part in SKIP_DIRS for part in parts):
            continue
        if rel.startswith(".git/"):
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            content = p.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable
        lang = detect_language(rel)
        out.append(
            RepoFile(
                path=rel,
                language=lang,
                content=content,
                symbols=extract_symbols(content, lang) if lang in CODE_LANGS else [],
                size=len(content),
            )
        )
    return out

"""Repo-level context retrieval — the capstone of RepoLens.

Instead of prompting the LLM with only the diff (single-file view), we:

1. Index every code file in the repo into overlapping chunks with
   extracted symbol names.
2. Build a query from the *changed* code: identifiers added/removed in the
   diff, plus symbols defined in changed files.
3. Rank chunks with BM25 (identifier-aware tokenization), boosting chunks
   that define a symbol referenced by the diff.
4. Assemble a context budget: changed-file full contents first, then the
   top cross-file chunks, de-duplicated and truncated to fit.

The result is a context pack the reviewer prompt uses, so comments can
reference how the change interacts with the rest of the repository.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import FileChange, RepoFile
from .repo_index import chunk_file

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_IDENT_FULL = re.compile(r"[a-z_][a-z0-9_]{2,}")

# identifiers too generic to drive retrieval
STOPWORDS = {
    "the", "and", "for", "with", "return", "returns", "this", "that", "self",
    "cls", "true", "false", "none", "null", "import", "from", "def", "class",
    "function", "const", "let", "var", "new", "not", "else", "elif", "then",
    "value", "values", "args", "kwargs", "params", "data", "result", "results",
    "item", "items", "index", "count", "string", "number", "list", "dict",
    "type", "types", "test", "tests", "main", "init", "setup", "should", "when", "given", "assert", "print", "console", "log",
}


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokenizer: splits camelCase/snake_case into parts
    and keeps the full identifier too, so `getUserById` matches `user`."""
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        if low in STOPWORDS:
            continue
        tokens.append(low)
        # split snake_case
        parts = [p for p in ident.split("_") if p]
        # split camelCase
        split_parts: list[str] = []
        for part in parts:
            split_parts.extend(re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", part))
        for sp in split_parts:
            sp = sp.lower()
            if len(sp) >= 3 and sp not in STOPWORDS and sp != low:
                tokens.append(sp)
    return tokens


class BM25Index:
    """Minimal Okapi BM25 over text chunks (pure stdlib)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, doc: dict, text: str, extra_tokens: list[str] | None = None) -> None:
        toks = tokenize(text) + (extra_tokens or [])
        self.docs.append(doc)
        self.doc_tokens.append(toks)
        freq = Counter(toks)
        self.doc_freqs.append(freq)
        self.doc_lens.append(len(toks))
        for term in freq:
            self.df[term] += 1

    def build(self) -> None:
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0

    def query(self, terms: list[str], top_k: int = 20) -> list[tuple[float, dict]]:
        if not self.docs:
            return []
        n = len(self.docs)
        scores = [0.0] * n
        qf = Counter(terms)
        for term, qweight in qf.items():
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in range(n):
                tf = self.doc_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * qweight * (tf * (self.k1 + 1)) / denom
        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s > 0),
            key=lambda x: -x[0],
        )
        return [(s, self.docs[i]) for s, i in ranked[:top_k]]


class RepoContextRetriever:
    """Builds the repo-level context pack for a review.

    Two retrieval signals are combined:
    1. **Lexical** — identifier-aware BM25 over chunks, queried with the
       identifiers the diff adds/removes.
    2. **Reference graph** — we compute which files *reference* symbols
       defined in the changed files (callers/importers). Those files are the
       most likely places a change breaks something, so their chunks get a
       score boost and an explicit "call graph" section is added to the
       context pack. This is what lets the reviewer catch cross-file
       regressions that no lexical query would surface.
    """

    #: multiplicative boost applied to chunks in files that reference a
    #: symbol defined by the changed files
    CALLER_BOOST = 2.0

    def __init__(self, repo_files: list[RepoFile], context_budget: int = 60_000) -> None:
        self.repo_files = {f.path: f for f in repo_files}
        self.context_budget = context_budget
        self.index = BM25Index()
        self._symbol_to_paths: dict[str, set[str]] = {}
        self._file_idents: dict[str, set[str]] = {}
        for f in repo_files:
            for chunk in chunk_file(f.path, f.content, f.language):
                self.index.add(chunk, chunk["text"], extra_tokens=[s.lower() for s in chunk["symbols"]])
            for sym in f.symbols:
                self._symbol_to_paths.setdefault(sym.lower(), set()).add(f.path)
            self._file_idents[f.path] = set(_IDENT_FULL.findall(f.content.lower()))
        self.index.build()
        # reference graph: file -> symbols defined ELSEWHERE that it mentions
        self._file_refs: dict[str, set[str]] = {}
        all_defined = set(self._symbol_to_paths)
        for f in repo_files:
            own = {s.lower() for s in f.symbols}
            self._file_refs[f.path] = (self._file_idents[f.path] & all_defined) - own

    # -- query construction -------------------------------------------------

    def diff_identifiers(self, changes: list[FileChange]) -> list[str]:
        """Identifiers introduced/removed by the diff — the retrieval query."""
        terms: list[str] = []
        for ch in changes:
            for h in ch.hunks:
                for ln in h.lines:
                    if ln.kind in ("add", "del"):
                        terms.extend(tokenize(ln.text))
        return terms

    def changed_symbols(self, changes: list[FileChange]) -> set[str]:
        syms: set[str] = set()
        for ch in changes:
            f = self.repo_files.get(ch.path)
            if f:
                syms.update(s.lower() for s in f.symbols)
        return syms

    def callers_of_changed_symbols(
        self, changes: list[FileChange]
    ) -> dict[str, list[str]]:
        """Reference graph: for each symbol defined in a changed file,
        the other files that reference it (callers/importers).

        Returns {symbol: [paths...]} restricted to symbols with >=1 caller.
        """
        changed_paths = {ch.path for ch in changes}
        out: dict[str, list[str]] = {}
        for ch in changes:
            f = self.repo_files.get(ch.path)
            if not f:
                continue
            for sym in f.symbols:
                sym_l = sym.lower()
                callers = sorted(
                    p
                    for p, refs in self._file_refs.items()
                    if sym_l in refs and p not in changed_paths
                )
                if callers:
                    out[sym] = callers
        return out

    # -- retrieval ----------------------------------------------------------

    def retrieve(
        self,
        changes: list[FileChange],
        top_k: int = 24,
    ) -> tuple[list[dict], list[str]]:
        """Return (ranked cross-file chunks, query terms used).

        Chunks belonging to the changed files themselves are excluded —
        the prompt already carries those files' full content.

        Scores combine BM25 lexical relevance with a reference-graph boost:
        chunks in files that reference a symbol defined by the changed files
        (callers/importers) are multiplied by CALLER_BOOST, because those are
        exactly the places a change is most likely to break.
        """
        changed_paths = {ch.path for ch in changes} | {
            ch.old_path for ch in changes if ch.old_path
        }
        terms = self.diff_identifiers(changes)
        # boost symbols defined in changed files: cross-file callers matter
        terms.extend(self.changed_symbols(changes))
        results = self.index.query(terms, top_k=top_k * 4)

        # files that call/reference symbols defined in the changed files
        caller_files: set[str] = set()
        for callers in self.callers_of_changed_symbols(changes).values():
            caller_files.update(callers)

        scored: list[tuple[float, dict]] = []
        for score, doc in results:
            if doc["path"] in changed_paths:
                continue
            boosted = score * self.CALLER_BOOST if doc["path"] in caller_files else score
            scored.append((boosted, doc))
        scored.sort(key=lambda x: -x[0])

        picked: list[dict] = []
        seen_spans: set[tuple[str, int, int]] = set()
        for score, doc in scored:
            key = (doc["path"], doc["start"], doc["end"])
            if key in seen_spans:
                continue
            seen_spans.add(key)
            picked.append({**doc, "score": round(score, 3), "is_caller": doc["path"] in caller_files})
            if len(picked) >= top_k:
                break
        return picked, terms

    # -- context pack assembly ----------------------------------------------

    def assemble_context(
        self,
        changes: list[FileChange],
        top_k: int = 24,
    ) -> tuple[str, list[str]]:
        """Assemble the context pack string + list of context file paths used.

        Layout:
          1. Full content of changed files (head version) — highest priority.
          2. CALL GRAPH section: symbols defined by the change and the files
             that reference them (reference-graph retrieval).
          3. Cross-file chunks ranked by BM25 + caller boost.
        Truncated to the configured character budget.
        """
        parts: list[str] = []
        used = 0
        context_files: list[str] = []

        # 1) changed files' full content
        for ch in changes:
            f = self.repo_files.get(ch.path)
            content = ch.new_content if ch.new_content is not None else (f.content if f else None)
            if content is None or ch.status == "deleted":
                continue
            block = f"### FILE (changed): {ch.path}\n```\n{content}\n```"
            if used + len(block) > self.context_budget:
                room = self.context_budget - used
                if room > 2000:
                    parts.append(block[:room] + "\n... (truncated)")
                    used = self.context_budget
                break
            parts.append(block)
            used += len(block)

        # 2) call graph: who references the changed symbols?
        callers = self.callers_of_changed_symbols(changes)
        if callers:
            lines = ["### CALL GRAPH (files referencing symbols defined by this change)"]
            for sym, paths in sorted(callers.items()):
                lines.append(f"- `{sym}` is referenced by: " + ", ".join(f"`{p}`" for p in paths[:6]))
            block = "\n".join(lines)
            if used + len(block) <= self.context_budget:
                parts.append(block)
                used += len(block)
                for paths in callers.values():
                    for p in paths:
                        if p not in context_files:
                            context_files.append(p)

        # 3) cross-file retrieval results
        chunks, _terms = self.retrieve(changes, top_k=top_k)
        for c in chunks:
            tag = " [CALLER]" if c.get("is_caller") else ""
            header = f"### REPO CONTEXT{tag}: {c['path']} (lines {c['start']}-{c['end']}, relevance {c['score']})"
            block = f"{header}\n```\n{c['text']}\n```"
            if used + len(block) > self.context_budget:
                break
            parts.append(block)
            used += len(block)
            if c["path"] not in context_files:
                context_files.append(c["path"])

        return "\n\n".join(parts), context_files

    def symbol_definitions(self, symbol: str) -> list[str]:
        """Paths where a symbol is defined (for comment grounding)."""
        return sorted(self._symbol_to_paths.get(symbol.lower(), set()))

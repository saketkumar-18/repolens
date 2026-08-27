# 🔍 RepoLens

**An LLM code-review agent with repo-level context retrieval.**

RepoLens reads a pull-request diff **plus the full repository**, retrieves the
cross-file context that matters for the changed lines, and produces
**line-level review comments** it can post back to GitHub as a formal review.

> **Capstone angle:** most LLM review tools prompt the model with the diff
> alone — a single-file view that misses cross-file breakage. RepoLens indexes
> the whole repo (symbols + BM25 over code chunks), builds a retrieval query
> from the identifiers the diff actually touches, and injects the most
> relevant surrounding code into the prompt. The reviewer can then catch bugs
> that are *invisible from the diff alone*.

---

## Example: a bug only repo context can catch

In [`saketkumar-18/repolens-demo` PR #1](https://github.com/saketkumar-18/repolens-demo/pull/1),
the diff changes `TaskStore.get()` to return a defensive `deepcopy`. The diff
itself looks fine — but the repo context shows `TaskService.complete_task()`
mutates the dict returned by `store.get()`:

```python
# src/taskman/service.py (UNCHANGED by the PR, retrieved as context)
def complete_task(self, task_id: int) -> dict:
    task = self.store.get(task_id)   # now returns a COPY
    task["done"] = True              # mutation silently lost!
    return task
```

RepoLens flags this as a **critical correctness bug** with a comment anchored
to the changed line in `store.py`, citing `service.py` as the affected caller.
A diff-only reviewer cannot see this at all.

---

## How it works

```
                 ┌──────────────────────────────────────────────┐
 PR diff ──────► │ 1. Diff parser: hunks + exact new-side lines │
                 └──────────────────┬───────────────────────────┘
                                    │
 ┌──────────────┐   ┌───────────────▼───────────────┐
 │ full repo    ├──►│ 2. Repo indexer                │
 │ (checkout or │   │    languages, symbols, chunks  │
 │  GitHub API) │   └───────────────┬───────────────┘
 └──────────────┘                   │
                 ┌──────────────────▼───────────────────────────┐
                 │ 3. Retrieval (the capstone)                   │
                 │    query = identifiers added/removed by diff  │
                 │            + symbols defined in changed files │
                 │    rank  = identifier-aware BM25 over chunks  │
                 │    pack  = changed files + top cross-file     │
                 │            chunks, within a char budget       │
                 └──────────────────┬───────────────────────────┘
                                    │
                 ┌──────────────────▼───────────────────────────┐
                 │ 4. Review                                     │
                 │    a. deterministic heuristics (secrets,      │
                 │       SQLi, debug prints, merge markers…)     │
                 │    b. LLM with repo-context prompt → JSON     │
                 │    c. validation: every comment line is       │
                 │       snapped to a real diff line; unknown    │
                 │       paths dropped; low confidence filtered  │
                 └──────────────────┬───────────────────────────┘
                                    │
                 ┌──────────────────▼───────────────────────────┐
                 │ 5. Output: markdown report, JSON, or a formal │
                 │    GitHub review with inline comments         │
                 └──────────────────────────────────────────────┘
```

### Design decisions

- **Zero-hallucination anchoring.** LLM-proposed comments are validated
  against the parsed diff: unknown file paths are dropped, line numbers are
  snapped to lines that actually exist in the hunk, so every posted comment
  anchors correctly on GitHub.
- **Two-signal retrieval.** Lexical (identifier-aware BM25) **plus a
  reference graph**: files that reference symbols defined by the changed
  files (callers/importers) get a 2× score boost and an explicit CALL GRAPH
  section in the prompt — those are exactly the places a change is most
  likely to break.
- **Heuristics + LLM, deduplicated.** A deterministic static pass catches the
  classic footguns (hardcoded secrets, f-string SQL, `except: pass`,
  breakpoints, merge markers) with no hallucination risk; the LLM adds
  semantic findings. On duplicate `(path, line)` the heuristic wins.
- **Batched reviews.** Large PRs are split into file batches (default 6 per
  LLM call), each with its own scoped context pack; a failed batch never
  blocks the rest, and verdicts merge conservatively (any request-changes
  wins).
- **Provider-agnostic LLM client.** Any OpenAI-compatible endpoint works
  (OpenRouter by default with a free-model fallback chain, OpenAI, Groq,
  Ollama, vLLM). Transient 429/5xx errors retry with backoff, then fall
  through to the next model. If the LLM is fully unavailable, you still get
  the heuristics review.
- **Pure-stdlib retrieval.** No tree-sitter / no vector DB: regex symbol
  extraction per language family + Okapi BM25 with camelCase/snake_case
  splitting. Installs anywhere, deterministic, fast enough for repos with
  thousands of files.
- **Per-repo config.** Drop a `.repolens.toml` in the repo root to tune
  budgets, pin a model, or focus/ignore paths (see the example in this repo).
- **Graceful degradation everywhere.** Binary files, lockfiles, deleted files
  and oversized PRs are skipped and reported, never fatal.

### GitHub Action (auto-review every PR)

This repo dogfoods itself: `.github/workflows/repolens-review.yml` runs
RepoLens on every PR and posts the review. To use it in any repo:

```yaml
# .github/workflows/repolens-review.yml  (copy from this repo)
# needs repo secret: OPENROUTER_API_KEY (or any supported provider key)
```

The action checks out the PR with full history, installs RepoLens, reviews
`base...head` against the local checkout, and posts the formal review with
inline comments using the built-in `GITHUB_TOKEN`.

---

## Install

```bash
git clone https://github.com/saketkumar-18/repolens
cd repolens
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # add your keys
repolens check         # verify configuration
```

Requirements: Python ≥ 3.10. Dependencies: `httpx`, `pydantic`, `fastapi`, `uvicorn`.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` (or `REPOLENS_API_KEY`, `OPENAI_API_KEY`) | LLM auth | — |
| `REPOLENS_BASE_URL` | OpenAI-compatible endpoint | `https://openrouter.ai/api/v1` |
| `REPOLENS_MODELS` | comma-separated fallback chain | free OpenRouter models |
| `GITHUB_TOKEN` | private repos + posting reviews | — |
| `REPOLENS_CONTEXT_BUDGET` | chars of repo context per review | `60000` |
| `REPOLENS_MAX_FILES` | max files sent to the LLM per PR | `10` |
| `REPOLENS_BATCH_SIZE` | files per LLM call for large PRs | `6` |

All of the above (plus `top_k_chunks`, `min_confidence`, `include_heuristics`,
`model`, and `[focus]` path globs) can also be set per-repo in a
`.repolens.toml` at the repo root — see the example in this repo.

## Usage

### Review a GitHub PR

```bash
# remote indexing via the GitHub API (no checkout needed)
repolens pr owner/repo 42

# faster + richer context from a local checkout
repolens pr owner/repo 42 --local ./checkout

# post the review back to GitHub (formal review + inline comments)
repolens pr owner/repo 42 --post

# see the exact API payload without posting
repolens pr owner/repo 42 --post --dry-run
```

### Review local git changes

```bash
repolens local . --base main            # review main...HEAD
repolens local . --diff changes.patch   # review an arbitrary patch file
repolens local . --json -o review.json  # machine-readable output
```

Exit codes: `0` approve/comment, `2` request-changes (CI-friendly), `1` error.

### HTTP API (for bots / webhooks)

```bash
repolens serve --host 0.0.0.0 --port 8787
```

```bash
curl -X POST localhost:8787/review/pr \
  -H 'Content-Type: application/json' \
  -d '{"repo": "owner/repo", "pr_number": 42, "post": false}'

# ad-hoc diff review with explicit file contents for context
curl -X POST localhost:8787/review/diff \
  -d '{"diff": "...", "files": {"app.py": "..."}}'

curl localhost:8787/reviews/<review_id>?fmt=markdown
```

Interactive docs at `/docs` (FastAPI/Swagger).

## Development

```bash
pytest            # 65 tests: parser, retrieval, reference graph, config,
                  # heuristics, batching, orchestrator, API
ruff check .      # lint
```

## Project layout

```
src/repolens/
├── models.py          # pydantic data model (diffs, comments, results)
├── diff_parser.py     # unified-diff parser, exact new-side line numbers
├── repo_index.py      # language detection, symbol extraction, chunking
├── retrieval.py       # identifier-aware BM25 + reference graph + context pack  ← capstone
├── config.py          # per-repo .repolens.toml configuration
├── heuristics.py      # deterministic static checks
├── llm.py             # OpenAI-compatible client, retries + fallback chain
├── prompts.py         # repo-aware reviewer prompt
├── reviewer.py        # orchestrator: diff → index → retrieve → review
├── github_client.py   # PR diff/tree fetch, review submission
├── poster.py          # ReviewResult → formal GitHub review
├── report.py          # markdown rendering
├── server.py          # FastAPI service
└── cli.py             # repolens CLI
```

## Limitations

- Symbol extraction is regex-based (no tree-sitter): recall is high for
  common definition forms but not exhaustive; a missed symbol only degrades
  retrieval ranking, never correctness.
- Free-tier LLM endpoints rate-limit; the fallback chain mitigates but does
  not eliminate 429s. For production use a paid key.
- Remote indexing fetches up to 400 code files from the head tree; very large
  monorepos should use `--local` with a checkout.

## License

MIT

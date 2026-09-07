# Changelog

## 0.1.0 — 2026-08-27

Initial release.

- Unified-diff parser with exact new-side line mapping
- Repo indexer: language detection, regex symbol extraction, chunking
- Two-signal retrieval: identifier-aware BM25 + reference-graph caller boost
- Deterministic heuristics pass (secrets, SQLi, debug prints, merge markers)
- OpenAI-compatible LLM client with retries + free-model fallback chain
- Batched reviews for large PRs
- Validated line-level comments anchored to real diff lines
- GitHub client: PR diff/tree fetch, blob caching, formal review submission
- Per-repo `.repolens.toml` configuration
- CLI (pr/local/serve/check) + FastAPI server + markdown/JSON reports
- GitHub Action for automatic PR reviews

<!-- test: trigger repolens-review -->

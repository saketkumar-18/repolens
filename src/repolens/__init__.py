"""RepoLens — LLM code-review agent with repo-level context retrieval.

Reads a PR diff plus the full repository, retrieves cross-file context
relevant to the changed lines (symbol-aware BM25), and produces
line-level review comments that can be posted back to GitHub.
"""

__version__ = "0.1.0"

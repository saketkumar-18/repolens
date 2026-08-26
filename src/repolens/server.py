"""FastAPI server: review PRs over HTTP (for CI bots / webhooks).

Endpoints:
  GET  /health                 — liveness + config state
  POST /review/pr              — {repo, pr_number, local_path?, post?, model?}
  POST /review/diff            — {repo_label?, base_ref?, head_ref?, diff, files?: {path: content}}
  GET  /reviews/{id}           — fetch a stored review result
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .diff_parser import parse_diff
from .github_client import GitHubClient, GitHubError
from .llm import LLMClient, LLMError
from .models import RepoFile, ReviewResult
from .poster import post_review
from .report import render_full_report
from .reviewer import ReviewConfig, Reviewer

_store: dict[str, dict[str, Any]] = {}
_store_lock = threading.Lock()


class PRReviewRequest(BaseModel):
    repo: str = Field(..., description="owner/name")
    pr_number: int
    local_path: str | None = None
    post: bool = False
    dry_run: bool = False
    model: str | None = None
    context_budget: int | None = None
    max_files: int | None = None


class DiffReviewRequest(BaseModel):
    diff: str
    repo_label: str = "adhoc"
    base_ref: str = "main"
    head_ref: str = "head"
    pr_title: str = ""
    pr_body: str = ""
    files: dict[str, str] = Field(default_factory=dict, description="path -> full content for context")
    model: str | None = None
    context_budget: int | None = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="RepoLens",
        version=__version__,
        description="LLM code-review agent with repo-level context retrieval.",
    )

    @app.get("/health")
    def health() -> dict:
        llm = LLMClient()
        return {
            "status": "ok",
            "version": __version__,
            "llm_configured": bool(llm.api_key),
            "llm_base_url": llm.base_url,
            "models": llm.models,
            "github_token": bool(os.getenv("GITHUB_TOKEN")),
        }

    @app.post("/review/pr")
    def review_pr(req: PRReviewRequest) -> dict:
        cfg = ReviewConfig()
        if req.model:
            cfg.model = req.model
        if req.context_budget:
            cfg.context_budget = req.context_budget
        if req.max_files:
            cfg.max_files = req.max_files
        reviewer = Reviewer(config=cfg)
        try:
            gh = GitHubClient()
            result = reviewer.review_pr(req.repo, req.pr_number, github=gh, local_path=req.local_path)
        except (GitHubError, LLMError) as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        review_id = uuid.uuid4().hex[:12]
        posted = None
        if req.post or req.dry_run:
            try:
                posted = post_review(result, req.repo, req.pr_number, github=gh, dry_run=req.dry_run)
            except GitHubError as e:
                raise HTTPException(status_code=502, detail=f"review done but posting failed: {e}") from e

        with _store_lock:
            _store[review_id] = {"result": result, "posted": posted}
        return {
            "review_id": review_id,
            "verdict": result.summary.verdict,
            "comments": len(result.comments),
            "blocking": result.blocking_count,
            "posted": posted,
            "result": result.model_dump(),
        }

    @app.post("/review/diff")
    def review_diff(req: DiffReviewRequest) -> dict:
        cfg = ReviewConfig()
        if req.model:
            cfg.model = req.model
        if req.context_budget:
            cfg.context_budget = req.context_budget
        reviewer = Reviewer(config=cfg)

        changes = parse_diff(req.diff)
        if not changes:
            raise HTTPException(status_code=400, detail="no file changes parsed from diff")

        from .repo_index import CODE_LANGS, detect_language, extract_symbols

        repo_files: list[RepoFile] = []
        for path, content in req.files.items():
            lang = detect_language(path)
            repo_files.append(
                RepoFile(
                    path=path,
                    language=lang,
                    content=content,
                    symbols=extract_symbols(content, lang) if lang in CODE_LANGS else [],
                    size=len(content),
                )
            )
        for ch in changes:
            if ch.path in req.files and ch.status != "deleted":
                ch.new_content = req.files[ch.path]

        result = reviewer._review(  # noqa: SLF001 — internal pipeline reuse
            changes=changes,
            repo_files=repo_files,
            repo_label=req.repo_label,
            base_ref=req.base_ref,
            head_ref=req.head_ref,
            pr_title=req.pr_title,
            pr_body=req.pr_body,
        )
        review_id = uuid.uuid4().hex[:12]
        with _store_lock:
            _store[review_id] = {"result": result, "posted": None}
        return {
            "review_id": review_id,
            "verdict": result.summary.verdict,
            "comments": len(result.comments),
            "blocking": result.blocking_count,
            "result": result.model_dump(),
        }

    @app.get("/reviews/{review_id}")
    def get_review(review_id: str, fmt: str = "json") -> Any:
        with _store_lock:
            entry = _store.get(review_id)
        if not entry:
            raise HTTPException(status_code=404, detail="review not found")
        result: ReviewResult = entry["result"]
        if fmt == "markdown":
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse(render_full_report(result))
        return {"posted": entry["posted"], "result": result.model_dump()}

    return app

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

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(_DEMO_PAGE)

    return app


_DEMO_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RepoLens — LLM Code Review</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background: #0b0e14; color: #e6e8ee; min-height: 100vh; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.5px; }
  h1 .lens { color: #38bdf8; }
  p.sub { color: #9aa3b2; margin: 0 0 28px; font-size: 15px; }
  .card { background: #11151f; border: 1px solid #232a3a; border-radius: 12px; padding: 20px; margin-bottom: 18px; }
  label { display: block; font-size: 13px; color: #9aa3b2; margin: 12px 0 6px; }
  input[type=text], select { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2b3346;
    background: #0b0e14; color: #e6e8ee; font-size: 14px; }
  input[type=text], select, textarea { border-color: #2b3346; }
  textarea { width: 100%; min-height: 180px; padding: 10px 12px; border-radius: 8px; border: 1px solid #2b3346;
    background: #0b0e14; color: #e6e8ee; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; resize: vertical; }
  button { margin-top: 16px; padding: 11px 22px; font-size: 14px; font-weight: 600; border: none; border-radius: 8px;
    background: linear-gradient(135deg, #38bdf8, #818cf8); color: #0b0e14; cursor: pointer; }
  button:disabled { opacity: 0.55; cursor: wait; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 200px; }
  .status { margin-top: 14px; font-size: 13px; color: #9aa3b2; }
  .verdict { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 13px; }
  .v-approve { background: #052e1a; color: #34d399; }
  .v-comment { background: #1e2a44; color: #93c5fd; }
  .v-block { background: #3d0f14; color: #f87171; }
  .cmt { border-left: 3px solid #2b3346; padding: 10px 14px; margin: 10px 0; background: #0d111a; border-radius: 0 8px 8px 0; }
  .cmt.block { border-left-color: #f87171; }
  .cmt .path { font-family: ui-monospace, monospace; font-size: 12.5px; color: #38bdf8; }
  .cmt .sev { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #9aa3b2; margin-left: 8px; }
  .cmt pre { white-space: pre-wrap; font-size: 13px; margin: 6px 0 0; color: #c9d1e0; }
  .cmt code { background: #0b0e14; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #fbbf24; }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: #9aa3b2; font-size: 12.5px; }
  a { color: #38bdf8; }
  .pill { display: inline-block; background: #0d111a; border: 1px solid #232a3a; border-radius: 999px;
    padding: 3px 10px; font-size: 11.5px; color: #9aa3b2; margin-right: 6px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Repo<span class="lens">Lens</span></h1>
  <p class="sub">LLM code-review agent with repo-level context retrieval. Paste a diff, get line-level review comments.
    <span class="pill">github.com/saketkumar-18/repolens</span></p>

  <div class="card">
    <div class="row">
      <div>
        <label for="repo">Repository (owner/name)</label>
        <input id="repo" type="text" value="saketkumar-18/repolens-demo" placeholder="owner/name">
      </div>
      <div>
        <label for="pr">Pull Request number</label>
        <input id="pr" type="text" value="1" placeholder="e.g. 1">
      </div>
    </div>
    <button id="go" onclick="runReview()">Review this PR</button>
    <div class="status" id="status"></div>
  </div>

  <div class="card">
    <label for="diff">…or paste a raw diff to review</label>
    <textarea id="diff" placeholder="diff --git a/app.py b/app.py&#10;--- a/app.py&#10;+++ b/app.py&#10;@@ -10,7 +10,7 @@&#10;- x = 1&#10;+ x = 2"></textarea>
    <button id="goDiff" onclick="runDiff()">Review diff</button>
  </div>

  <div id="out"></div>
</div>
<script>
async function runReview() {
  const repo = document.getElementById('repo').value.trim();
  const pr = parseInt(document.getElementById('pr').value.trim(), 10);
  if (!repo || !pr) { setStatus('Enter a repo (owner/name) and PR number.'); return; }
  await go('/review/pr', { repo, pr_number: pr }, `Reviewing ${repo}#${pr} — fetching diff, indexing repo, running LLM review… (10-60s)`);
}
async function runDiff() {
  const diff = document.getElementById('diff').value;
  if (!diff.trim()) { setStatus('Paste a diff first.'); return; }
  await go('/review/diff', { diff, repo_label: 'adhoc' }, 'Reviewing pasted diff…');
}
async function go(path, body, msg) {
  const btns = document.querySelectorAll('button');
  btns.forEach(b => b.disabled = true);
  setStatus(msg);
  document.getElementById('out').innerHTML = '';
  try {
    const r = await fetch(path, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) { setStatus('Error: ' + (data.detail || JSON.stringify(data)).slice(0, 300)); return; }
    setStatus(`Done — verdict: ${data.verdict} · ${data.comments} comments · ${data.blocking} blocking`);
    render(data.result);
  } catch (e) { setStatus('Network error: ' + e.message); }
  finally { btns.forEach(b => b.disabled = false); }
}
function render(res) {
  const out = document.getElementById('out');
  const v = res.summary.verdict || 'COMMENT';
  const cls = v.toLowerCase().includes('block') ? 'v-block' : (v.toLowerCase().includes('approve') ? 'v-approve' : 'v-comment');
  let html = `<div class="card"><span class="verdict ${cls}">${v}</span>`;
  if (res.summary && res.summary.summary_text) html += `<div style="margin-top:10px">${esc(res.summary.summary_text)}</div>`;
  html += `</div>`;
  for (const c of (res.comments || [])) {
    html += `<div class="cmt ${c.severity==='blocker'?'block':''}"><span class="path">${esc(c.path)}${c.line ? ':' + c.line : ''}</span><span class="sev">${esc(c.severity || '')}</span><pre>${esc(c.comment)}</pre></div>`;
  }
  out.innerHTML = html;
}
function esc(s) { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }
function setStatus(s) { document.getElementById('status').textContent = s; }
</script>
</body>
</html>
"""

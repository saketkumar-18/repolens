"""The reviewer orchestrator: diff -> index -> retrieve -> LLM -> validated comments."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from .config import apply_config, filter_paths, load_repo_config
from .diff_parser import parse_diff, validate_comment_line
from .github_client import GitHubClient
from .heuristics import run_heuristics
from .llm import LLMClient, LLMError
from .models import FileChange, ReviewComment, ReviewResult, ReviewSummary, Severity
from .prompts import build_review_prompt
from .repo_index import CODE_LANGS, detect_language, iter_repo_files
from .retrieval import RepoContextRetriever

# files never worth sending to the LLM
SKIP_SUFFIXES = {
    ".min.js", ".min.css", ".map", ".lock", ".sum", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot", ".svg", ".pdf", ".zip",
    ".tar", ".gz", ".whl", ".pyc", ".class", ".o", ".so", ".dll", ".exe",
}
SKIP_EXACT = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "composer.lock", "Cargo.lock", "go.sum", "uv.lock",
}


def _log(msg: str) -> None:
    print(f"[repolens] {msg}", file=sys.stderr, flush=True)


CONFIG_NOTE = ".repolens.toml"


@dataclass
class ReviewConfig:
    context_budget: int = int(os.getenv("REPOLENS_CONTEXT_BUDGET", "60000"))
    max_files: int = int(os.getenv("REPOLENS_MAX_FILES", "10"))
    top_k_chunks: int = 24
    min_confidence: float = 0.4
    include_heuristics: bool = True
    batch_size: int = int(os.getenv("REPOLENS_BATCH_SIZE", "6"))
    model: str | None = None  # pin a model; None = fallback chain
    focus_paths: list[str] = field(default_factory=list)
    ignore_paths: list[str] = field(default_factory=list)


@dataclass
class Reviewer:
    config: ReviewConfig = field(default_factory=ReviewConfig)
    llm: LLMClient = field(default_factory=LLMClient)

    # ------------------------------------------------------------------
    # local repo review
    # ------------------------------------------------------------------

    def review_local(
        self,
        repo_path: str,
        diff_text: str | None = None,
        base_ref: str = "main",
        head_ref: str = "HEAD",
        pr_title: str = "",
        pr_body: str = "",
    ) -> ReviewResult:
        """Review a local git repo. Diff defaults to base...HEAD."""
        import subprocess

        if diff_text is None:
            diff_text = subprocess.run(
                ["git", "diff", f"{base_ref}...{head_ref}"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        try:
            head_sha = subprocess.run(
                ["git", "rev-parse", head_ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            head_sha = ""  # not a git repo / bad ref — review still works

        changes = parse_diff(diff_text)
        _log(f"parsed diff: {len(changes)} changed files")

        # attach head-version content for changed files
        for ch in changes:
            fp = os.path.join(repo_path, ch.path.replace("/", os.sep))
            if os.path.isfile(fp) and ch.status != "deleted":
                try:
                    with open(fp, encoding="utf-8") as f:
                        ch.new_content = f.read()
                except (OSError, UnicodeDecodeError):
                    pass

        repo_files = iter_repo_files(repo_path)
        _log(f"indexed repo: {len(repo_files)} files")

        # per-repo config (.repolens.toml) — CLI flags already set win only if
        # explicitly provided; repo config fills the rest
        repo_cfg = load_repo_config(repo_path)
        if repo_cfg:
            apply_config(self.config, repo_cfg)
            focus = repo_cfg.get("focus") or {}
            self.config.focus_paths = focus.get("paths") or self.config.focus_paths
            self.config.ignore_paths = focus.get("ignore") or self.config.ignore_paths
            _log(f"loaded {CONFIG_NOTE} from repo")

        return self._review(
            changes=changes,
            repo_files=repo_files,
            repo_label=os.path.basename(os.path.abspath(repo_path)),
            base_ref=base_ref,
            head_ref=head_ref,
            head_sha=head_sha,
            pr_title=pr_title,
            pr_body=pr_body,
        )

    # ------------------------------------------------------------------
    # GitHub PR review
    # ------------------------------------------------------------------

    def review_pr(
        self,
        repo: str,
        pr_number: int,
        github: GitHubClient | None = None,
        local_path: str | None = None,
    ) -> ReviewResult:
        """Review a GitHub PR. If local_path is given, index the local checkout
        (faster, full files); otherwise fetch the head tree via the API."""
        gh = github or GitHubClient()
        pr = gh.get_pr(repo, pr_number)
        diff_text = gh.get_pr_diff(repo, pr_number)
        changes = parse_diff(diff_text)
        _log(f"PR #{pr_number}: {len(changes)} changed files")

        head_ref = pr["head"]["ref"]
        base_ref = pr["base"]["ref"]
        head_sha = pr["head"]["sha"]

        if local_path:
            repo_files = iter_repo_files(local_path)
            for ch in changes:
                fp = os.path.join(local_path, ch.path.replace("/", os.sep))
                if os.path.isfile(fp) and ch.status != "deleted":
                    try:
                        with open(fp, encoding="utf-8") as f:
                            ch.new_content = f.read()
                    except (OSError, UnicodeDecodeError):
                        pass
            _log(f"indexed local checkout: {len(repo_files)} files")
        else:
            entries = gh.list_tree(repo, head_sha)
            # fetch only code-ish files, capped
            code_entries = [
                e
                for e in entries
                if detect_language(e["path"]) in CODE_LANGS
                or detect_language(e["path"]) in ("yaml", "toml", "json", "docker", "sql")
            ][:400]
            blob_shas = {e["path"]: e["sha"] for e in code_entries}
            repo_files = gh.fetch_repo_files(
                repo, head_sha, [e["path"] for e in code_entries], blob_shas=blob_shas
            )
            file_map = {f.path: f for f in repo_files}
            for ch in changes:
                f = file_map.get(ch.path)
                if f and ch.status != "deleted":
                    ch.new_content = f.content
            _log(f"indexed remote tree: {len(repo_files)} files")

        result = self._review(
            changes=changes,
            repo_files=repo_files,
            repo_label=repo,
            base_ref=base_ref,
            head_ref=head_ref,
            head_sha=head_sha,
            pr_number=pr_number,
            pr_title=pr.get("title", ""),
            pr_body=pr.get("body") or "",
        )
        return result

    # ------------------------------------------------------------------
    # core pipeline
    # ------------------------------------------------------------------

    def _review(
        self,
        changes: list[FileChange],
        repo_files,
        repo_label: str,
        base_ref: str,
        head_ref: str,
        head_sha: str = "",
        pr_number: int | None = None,
        pr_title: str = "",
        pr_body: str = "",
    ) -> ReviewResult:
        self._llm_summary = {}

        # 1) filter to reviewable files
        reviewable: list[FileChange] = []
        skipped: list[str] = []
        for ch in changes:
            if ch.is_binary or ch.path in SKIP_EXACT or any(
                ch.path.endswith(s) for s in SKIP_SUFFIXES
            ):
                skipped.append(ch.path)
                continue
            if ch.status == "deleted":
                skipped.append(ch.path)  # nothing to annotate on a deleted file
                continue
            reviewable.append(ch)

        # focus/ignore globs from repo config
        if self.config.focus_paths or self.config.ignore_paths:
            kept = set(
                filter_paths(
                    [c.path for c in reviewable],
                    {"paths": self.config.focus_paths, "ignore": self.config.ignore_paths},
                )
            )
            dropped = [c for c in reviewable if c.path not in kept]
            reviewable = [c for c in reviewable if c.path in kept]
            skipped.extend(c.path for c in dropped)

        # cap LLM workload: biggest diffs first
        reviewable.sort(key=lambda c: c.additions + c.deletions, reverse=True)
        if len(reviewable) > self.config.max_files:
            skipped.extend(c.path for c in reviewable[self.config.max_files :])
            reviewable = reviewable[: self.config.max_files]

        result = ReviewResult(
            repo=repo_label,
            pr_number=pr_number,
            base_ref=base_ref,
            head_ref=head_ref,
            head_sha=head_sha,
            files_reviewed=len(reviewable),
            files_skipped=skipped,
        )
        if not reviewable:
            result.summary = ReviewSummary(verdict="approve", overview="No reviewable code changes.")
            return result

        # 2) heuristics pass (deterministic)
        heuristic_comments = run_heuristics(reviewable) if self.config.include_heuristics else []

        # 3) repo-level context retrieval (BM25 + reference graph)
        retriever = RepoContextRetriever(repo_files, context_budget=self.config.context_budget)
        result.context_files_used = []

        # 4) LLM review — batched for large PRs
        llm_comments: list[ReviewComment] = []
        models_used: list[str] = []
        total_tokens = 0
        batch_size = max(1, self.config.batch_size)
        batches = [
            reviewable[i : i + batch_size] for i in range(0, len(reviewable), batch_size)
        ]
        for bi, batch in enumerate(batches, 1):
            # per-batch context: changed files of this batch + repo retrieval
            # scoped to the batch's identifiers
            context_pack, context_files = retriever.assemble_context(
                batch, top_k=self.config.top_k_chunks
            )
            for p in context_files:
                if p not in result.context_files_used:
                    result.context_files_used.append(p)
            _log(
                f"batch {bi}/{len(batches)}: {len(batch)} file(s), "
                f"context {len(context_pack)} chars from {len(context_files)} repo files"
            )
            messages = build_review_prompt(
                batch,
                context_pack,
                pr_title=pr_title,
                pr_body=pr_body,
                base_ref=base_ref,
                head_ref=head_ref,
            )
            try:
                data, model_used, tokens = self.llm.chat_json(
                    messages, max_tokens=4096, model=self.config.model
                )
                models_used.append(model_used)
                total_tokens += tokens
                batch_comments = self._parse_llm_review(data, batch)
                llm_comments.extend(batch_comments)
                _log(f"LLM ({model_used}): {len(batch_comments)} comments, {tokens} tokens")
            except LLMError as e:
                _log(f"LLM review failed for batch {bi} ({e}); continuing with remaining batches")

        result.tokens_used = total_tokens
        if models_used:
            result.model = models_used[0] if len(set(models_used)) == 1 else ", ".join(sorted(set(models_used)))
        else:
            _log("LLM unavailable for all batches; returning heuristics-only review")
            result.model = "none (LLM unavailable)"

        # 5) merge + dedupe
        result.comments = self._merge_comments(heuristic_comments, llm_comments)

        # 6) summary / verdict
        result.summary = self._final_summary(
            result, had_llm=bool(result.model and "none" not in result.model)
        )
        return result

    def _parse_llm_review(self, data: dict, changes: list[FileChange]) -> list[ReviewComment]:
        """Validate + normalize LLM output against the actual diff."""
        change_map = {c.path: c for c in changes}
        valid_paths = set(change_map)
        comments: list[ReviewComment] = []
        for raw in data.get("comments", []) or []:
            if not isinstance(raw, dict):
                continue
            path = raw.get("path", "")
            if path not in valid_paths:
                _log(f"dropping comment for unknown path: {path!r}")
                continue
            try:
                sev = Severity(str(raw.get("severity", "suggestion")).lower())
            except ValueError:
                sev = Severity.SUGGESTION
            line = raw.get("line")
            line = int(line) if isinstance(line, (int, float)) else None
            line = validate_comment_line(change_map[path], line)
            try:
                conf = float(raw.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            conf = max(0.0, min(1.0, conf))
            if conf < self.config.min_confidence and sev not in (Severity.CRITICAL,):
                continue
            msg = str(raw.get("message", "")).strip()
            if not msg:
                continue
            comments.append(
                ReviewComment(
                    path=path,
                    line=line,
                    severity=sev,
                    category=str(raw.get("category", "general")),
                    message=msg[:2000],
                    suggestion=(str(raw["suggestion"])[:2000] if raw.get("suggestion") else None),
                    source="llm",
                    confidence=conf,
                    context_refs=[str(r) for r in (raw.get("context_refs") or [])][:5],
                )
            )
        # accumulate summary across batches (first verdict/overview wins,
        # strengths are merged)
        batch_summary = data.get("summary") or {}
        if not self._llm_summary:
            self._llm_summary = {
                "verdict": batch_summary.get("verdict", ""),
                "overview": batch_summary.get("overview", ""),
                "strengths": list(batch_summary.get("strengths") or []),
            }
        else:
            self._llm_summary["strengths"].extend(batch_summary.get("strengths") or [])
            if str(batch_summary.get("verdict", "")).lower() == "request_changes":
                self._llm_summary["verdict"] = "request_changes"
        return comments

    def _merge_comments(
        self, heuristics: list[ReviewComment], llm: list[ReviewComment]
    ) -> list[ReviewComment]:
        """Heuristics win on duplicate (path, line) — they're deterministic."""
        seen: set[tuple[str, int | None]] = set()
        merged: list[ReviewComment] = []
        for c in heuristics:
            key = (c.path, c.line)
            seen.add(key)
            merged.append(c)
        for c in llm:
            key = (c.path, c.line)
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
        order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.SUGGESTION: 2, Severity.PRAISE: 3}
        merged.sort(key=lambda c: (order[c.severity], c.path, c.line or 0))
        return merged

    def _final_summary(self, result: ReviewResult, had_llm: bool) -> ReviewSummary:
        llm_sum = getattr(self, "_llm_summary", None) or {}
        blocking = result.blocking_count
        if blocking > 0:
            verdict = "request_changes"
        elif had_llm and str(llm_sum.get("verdict", "")).lower() in ("approve", "request_changes", "comment"):
            verdict = llm_sum["verdict"].lower()
        else:
            verdict = "approve" if blocking == 0 else "request_changes"
        overview = str(llm_sum.get("overview", "")).strip() if had_llm else ""
        if not overview:
            overview = (
                f"Reviewed {result.files_reviewed} file(s); "
                f"{len(result.comments)} comment(s), {blocking} blocking."
            )
        strengths = [str(s) for s in (llm_sum.get("strengths") or [])][:5] if had_llm else []
        return ReviewSummary(verdict=verdict, overview=overview, strengths=strengths)

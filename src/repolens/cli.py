"""RepoLens command-line interface.

Examples:
  # Review a GitHub PR (remote indexing via API)
  repolens pr saketkumar-18/myrepo 42

  # Review a GitHub PR using a local checkout for context
  repolens pr saketkumar-18/myrepo 42 --local ./myrepo

  # Post the review back to GitHub
  repolens pr saketkumar-18/myrepo 42 --post

  # Review local git changes (main...HEAD)
  repolens local . --base main

  # Review an arbitrary diff file against a repo
  repolens local . --diff changes.patch
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .github_client import GitHubClient, GitHubError
from .llm import LLMError
from .models import ReviewResult
from .poster import post_review
from .report import render_full_report
from .reviewer import ReviewConfig, Reviewer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repolens",
        description="LLM code-review agent with repo-level context retrieval.",
    )
    p.add_argument("--version", action="version", version=f"repolens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", help="pin a specific LLM model (default: fallback chain)")
        sp.add_argument("--context-budget", type=int, default=None, help="chars of repo context (default 60000)")
        sp.add_argument("--max-files", type=int, default=None, help="max files sent to the LLM (default 10)")
        sp.add_argument("--no-heuristics", action="store_true", help="skip the static heuristics pass")
        sp.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
        sp.add_argument("--output", "-o", help="write the report to a file")

    pr = sub.add_parser("pr", help="review a GitHub pull request")
    pr.add_argument("repo", help="owner/name")
    pr.add_argument("number", type=int, help="PR number")
    pr.add_argument("--local", help="path to a local checkout of the repo (better context, faster)")
    pr.add_argument("--post", action="store_true", help="post the review back to GitHub")
    pr.add_argument("--dry-run", action="store_true", help="with --post: print payload, don't post")
    add_common(pr)

    loc = sub.add_parser("local", help="review local git changes")
    loc.add_argument("path", nargs="?", default=".", help="repo path (default: cwd)")
    loc.add_argument("--base", default="main", help="base ref (default: main)")
    loc.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    loc.add_argument("--diff", help="read this diff/patch file instead of running git diff")
    loc.add_argument("--title", default="", help="PR title for prompt context")
    loc.add_argument("--body", default="", help="PR description for prompt context")
    add_common(loc)

    srv = sub.add_parser("serve", help="run the HTTP API server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8787)
    sub.add_parser("check", help="verify configuration (API key, GitHub token)")

    return p


def _make_config(args: argparse.Namespace) -> ReviewConfig:
    cfg = ReviewConfig()
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "context_budget", None):
        cfg.context_budget = args.context_budget
    if getattr(args, "max_files", None):
        cfg.max_files = args.max_files
    if getattr(args, "no_heuristics", False):
        cfg.include_heuristics = False
    return cfg


def _emit(result: ReviewResult, args: argparse.Namespace) -> None:
    if args.json:
        text = result.model_dump_json(indent=2)
    else:
        text = render_full_report(result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[repolens] report written to {args.output}", file=sys.stderr)
    else:
        print(text)


def cmd_check() -> int:
    from .llm import LLMClient

    ok = True
    llm = LLMClient()
    if llm.api_key:
        print(f"✓ LLM API key present ({llm.api_key[:10]}…)  base={llm.base_url}")
        print(f"  model chain: {' -> '.join(llm.models)}")
    else:
        print("✗ No LLM API key. Set OPENROUTER_API_KEY (or REPOLENS_API_KEY).")
        ok = False
    gh = GitHubClient()
    if gh.token:
        print(f"✓ GITHUB_TOKEN present ({gh.token[:8]}…)")
    else:
        print("⚠ No GITHUB_TOKEN — public repos work read-only; --post will fail.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "check":
        return cmd_check()

    if args.command == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    reviewer = Reviewer(config=_make_config(args))

    try:
        if args.command == "pr":
            gh = GitHubClient()
            result = reviewer.review_pr(args.repo, args.number, github=gh, local_path=args.local)
            _emit(result, args)
            if args.post or args.dry_run:
                outcome = post_review(result, args.repo, args.number, github=gh, dry_run=args.dry_run)
                if args.dry_run:
                    print(json.dumps(outcome, indent=2)[:4000])
                else:
                    print(f"[repolens] posted review id={outcome.get('review_id')} state={outcome.get('state')}")
            return 0 if result.summary.verdict != "request_changes" else 2
        elif args.command == "local":
            diff_text = None
            if args.diff:
                with open(args.diff, encoding="utf-8") as f:
                    diff_text = f.read()
            result = reviewer.review_local(
                args.path,
                diff_text=diff_text,
                base_ref=args.base,
                head_ref=args.head,
                pr_title=args.title,
                pr_body=args.body,
            )
            _emit(result, args)
            return 0 if result.summary.verdict != "request_changes" else 2
    except (GitHubError, LLMError, OSError) as e:
        print(f"[repolens] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

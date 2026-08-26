"""Post a ReviewResult back to GitHub as a formal review + summary comment."""

from __future__ import annotations

import sys

from .github_client import GitHubClient, GitHubError
from .models import ReviewResult
from .report import render_comment_body, render_summary_comment

_VERDICT_EVENT = {
    "approve": "APPROVE",
    "request_changes": "REQUEST_CHANGES",
    "comment": "COMMENT",
}


def post_review(
    result: ReviewResult,
    repo: str,
    pr_number: int,
    github: GitHubClient | None = None,
    dry_run: bool = False,
) -> dict:
    """Submit the review. Returns {'review_id':..., 'comment_id':...} or dry-run payload."""
    if not result.head_sha:
        raise GitHubError("result has no head_sha; cannot post review")

    inline = []
    for c in result.comments:
        if c.line is None:
            continue  # file-level comments go into the summary
        inline.append(
            {
                "path": c.path,
                "line": c.line,
                "side": "RIGHT",
                "body": render_comment_body(c),
            }
        )

    event = _VERDICT_EVENT[result.summary.verdict]
    summary_body = render_summary_comment(result)

    if dry_run:
        return {
            "dry_run": True,
            "event": event,
            "commit_id": result.head_sha,
            "body": summary_body,
            "comments": inline,
        }

    gh = github or GitHubClient()
    try:
        review = gh.submit_review(
            repo=repo,
            pr_number=pr_number,
            commit_id=result.head_sha,
            event=event,
            body=summary_body,
            comments=inline,
        )
    except GitHubError as e:
        # Common failures: REQUEST_CHANGES on your own PR, or a comment line
        # not in the diff (force-push race). Escalate gracefully:
        # 1) retry as COMMENT keeping inline comments
        # 2) if that fails, summary-only
        print(f"[repolens] full review failed ({e}); retrying as COMMENT with inline comments", file=sys.stderr)
        try:
            review = gh.submit_review(
                repo=repo,
                pr_number=pr_number,
                commit_id=result.head_sha,
                event="COMMENT",
                body=summary_body,
                comments=inline,
            )
        except GitHubError as e2:
            print(f"[repolens] inline comments rejected ({e2}); posting summary only", file=sys.stderr)
            review = gh.submit_review(
                repo=repo,
                pr_number=pr_number,
                commit_id=result.head_sha,
                event="COMMENT",
                body=summary_body + "\n\n_(inline comments could not be anchored to the diff)_",
                comments=[],
            )
    return {"review_id": review.get("id"), "state": review.get("state")}

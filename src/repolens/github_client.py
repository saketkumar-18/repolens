"""GitHub REST client: fetch PR diffs + head-tree files, post reviews.

Uses httpx directly (no PyGithub) with pagination and polite error handling.
"""

from __future__ import annotations

import base64
import os

import httpx

from .models import RepoFile

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


def _resolve_token() -> str:
    """GITHUB_TOKEN env var first, then fall back to `gh auth token`."""
    tok = os.getenv("GITHUB_TOKEN") or ""
    if tok:
        return tok
    try:
        import subprocess

        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token or _resolve_token()
        self._blob_cache: dict[str, str] = {}  # blob sha -> content
        self._client = httpx.Client(
            timeout=60.0,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, **params) -> httpx.Response:
        url = path if path.startswith("http") else f"{API}{path}"
        resp = self._client.get(url, params=params or None)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {url} -> {resp.status_code}: {resp.text[:300]}")
        return resp

    def _paginate(self, path: str, per_page: int = 100, **params) -> list[dict]:
        items: list[dict] = []
        page = 1
        while True:
            resp = self._get(path, per_page=per_page, page=page, **params)
            batch = resp.json()
            if not isinstance(batch, list):
                raise GitHubError(f"expected list from {path}, got {type(batch)}")
            items.extend(batch)
            if len(batch) < per_page:
                return items
            page += 1

    # ------------------------------------------------------------------
    # PR fetching
    # ------------------------------------------------------------------

    def get_pr(self, repo: str, pr_number: int) -> dict:
        return self._get(f"/repos/{repo}/pulls/{pr_number}").json()

    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        url = f"{API}/repos/{repo}/pulls/{pr_number}"
        resp = self._client.get(url, headers={"Accept": "application/vnd.github.diff"})
        if resp.status_code >= 400:
            raise GitHubError(f"diff fetch failed: {resp.status_code}: {resp.text[:300]}")
        return resp.text

    def list_pr_files(self, repo: str, pr_number: int) -> list[dict]:
        return self._paginate(f"/repos/{repo}/pulls/{pr_number}/files")

    def get_file_at_ref(self, repo: str, path: str, ref: str) -> str | None:
        """Fetch a file's content at a given ref. None if missing/too big/binary."""
        try:
            resp = self._get(f"/repos/{repo}/contents/{path}", ref=ref)
        except GitHubError:
            return None
        data = resp.json()
        if isinstance(data, list) or data.get("encoding") != "base64":
            return None
        if data.get("size", 0) > 400_000:
            return None
        try:
            return base64.b64decode(data.get("content", "")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None

    def list_tree(self, repo: str, ref: str, max_files: int = 2000) -> list[dict]:
        """Recursive tree listing at ref. Returns [{path, sha, size}] for blobs."""
        resp = self._get(f"/repos/{repo}/git/trees/{ref}", recursive="1")
        tree = resp.json().get("tree", [])
        entries: list[dict] = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            if entry.get("size", 0) > 400_000:
                continue
            entries.append(
                {"path": entry["path"], "sha": entry.get("sha", ""), "size": entry.get("size", 0)}
            )
            if len(entries) >= max_files:
                break
        return entries

    def get_blob(self, repo: str, blob_sha: str) -> str | None:
        """Fetch a blob by sha (cheaper + cacheable vs contents endpoint)."""
        try:
            resp = self._get(f"/repos/{repo}/git/blobs/{blob_sha}")
        except GitHubError:
            return None
        data = resp.json()
        if data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data.get("content", "")).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return None

    def fetch_repo_files(
        self,
        repo: str,
        ref: str,
        paths: list[str],
        blob_shas: dict[str, str] | None = None,
    ) -> list[RepoFile]:
        """Fetch file contents for remote indexing.

        When ``blob_shas`` maps path->sha, unchanged blobs are served from the
        in-process cache (keyed by blob sha), so repeated reviews of the same
        repo only download what changed.
        """
        from .repo_index import CODE_LANGS, detect_language, extract_symbols

        out: list[RepoFile] = []
        for p in paths:
            sha = (blob_shas or {}).get(p, "")
            content: str | None = None
            if sha and sha in self._blob_cache:
                content = self._blob_cache[sha]
            elif sha:
                content = self.get_blob(repo, sha)
                if content is not None:
                    self._blob_cache[sha] = content
            if content is None:
                content = self.get_file_at_ref(repo, p, ref)
            if content is None:
                continue
            lang = detect_language(p)
            out.append(
                RepoFile(
                    path=p,
                    language=lang,
                    content=content,
                    symbols=extract_symbols(content, lang) if lang in CODE_LANGS else [],
                    size=len(content),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Review posting
    # ------------------------------------------------------------------

    def submit_review(
        self,
        repo: str,
        pr_number: int,
        commit_id: str,
        event: str,
        body: str,
        comments: list[dict],
    ) -> dict:
        """Submit a formal review with inline comments atomically.

        event: APPROVE | REQUEST_CHANGES | COMMENT
        comments: [{path, line, side, body}]
        """
        payload: dict = {
            "commit_id": commit_id,
            "event": event,
            "body": body,
        }
        if comments:
            payload["comments"] = comments
        resp = self._client.post(f"{API}/repos/{repo}/pulls/{pr_number}/reviews", json=payload)
        if resp.status_code >= 400:
            raise GitHubError(f"review submit failed: {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def post_issue_comment(self, repo: str, pr_number: int, body: str) -> dict:
        resp = self._client.post(
            f"{API}/repos/{repo}/issues/{pr_number}/comments", json={"body": body}
        )
        if resp.status_code >= 400:
            raise GitHubError(f"comment failed: {resp.status_code}: {resp.text[:300]}")
        return resp.json()

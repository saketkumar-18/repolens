"""API server tests (FastAPI TestClient, stubbed LLM)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from repolens import server as server_mod
from repolens.reviewer import Reviewer

DIFF = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1,2 @@
 x = 1
+print("debug")
"""

PAYLOAD = {
    "summary": {"verdict": "comment", "overview": "minor", "strengths": []},
    "comments": [
        {
            "path": "app.py",
            "line": 2,
            "severity": "suggestion",
            "category": "style",
            "message": "remove debug print",
            "confidence": 0.8,
        }
    ],
}


class StubLLM:
    def chat_json(self, messages, **kwargs):
        return PAYLOAD, "stub", 10


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server_mod, "Reviewer", lambda config=None: Reviewer(config=config, llm=StubLLM()))  # type: ignore[arg-type]
    app = server_mod.create_app()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models" in body


def test_review_diff_endpoint(client):
    r = client.post(
        "/review/diff",
        json={
            "diff": DIFF,
            "repo_label": "demo",
            "files": {"app.py": "x = 1\nprint('debug')\n"},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["comments"] >= 1
    assert body["result"]["files_reviewed"] == 1
    review_id = body["review_id"]

    r2 = client.get(f"/reviews/{review_id}")
    assert r2.status_code == 200
    r3 = client.get(f"/reviews/{review_id}", params={"fmt": "markdown"})
    assert r3.status_code == 200
    assert "RepoLens" in r3.text


def test_review_diff_rejects_empty(client):
    r = client.post("/review/diff", json={"diff": "not a diff"})
    assert r.status_code == 400


def test_review_missing_404(client):
    assert client.get("/reviews/nope").status_code == 404

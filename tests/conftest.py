"""Shared fixtures: a small synthetic repo + a realistic diff."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic repo: a tiny user service with a bug planted for retrieval tests
# ---------------------------------------------------------------------------

REPO_FILES = {
    "src/service/user_service.py": textwrap.dedent(
        '''\
        """User service."""
        from src.repo.db import get_connection


        class UserService:
            def __init__(self):
                self.conn = get_connection()

            def get_user(self, user_id: int):
                cur = self.conn.cursor()
                cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                return cur.fetchone()

            def deactivate_user(self, user_id: int):
                cur = self.conn.cursor()
                cur.execute("UPDATE users SET active = 0 WHERE id = %s", (user_id,))
                self.conn.commit()
        '''
    ),
    "src/repo/db.py": textwrap.dedent(
        '''\
        """DB helpers."""
        import sqlite3


        def get_connection():
            return sqlite3.connect(":memory:")


        def close_connection(conn):
            conn.close()
        '''
    ),
    "src/api/routes.py": textwrap.dedent(
        '''\
        """API routes."""
        from src.service.user_service import UserService

        svc = UserService()


        def handle_get_user(request):
            user_id = int(request["user_id"])
            return svc.get_user(user_id)
        '''
    ),
    "src/utils/validators.py": textwrap.dedent(
        '''\
        """Validators."""


        def validate_email(email: str) -> bool:
            return "@" in email and "." in email.split("@")[-1]


        def validate_age(age) -> bool:
            return isinstance(age, int) and 0 < age < 150
        '''
    ),
    "tests/test_user_service.py": textwrap.dedent(
        '''\
        from src.service.user_service import UserService


        def test_get_user():
            svc = UserService()
            assert svc.get_user(1) is None
        '''
    ),
    "README.md": "# demo repo\n",
}

# A PR that adds a delete_user method with a SQL injection bug + a debug print,
# and modifies validators.
SAMPLE_DIFF = textwrap.dedent(
    '''\
    diff --git a/src/service/user_service.py b/src/service/user_service.py
    index 1111111..2222222 100644
    --- a/src/service/user_service.py
    +++ b/src/service/user_service.py
    @@ -14,3 +14,12 @@ class UserService:
         def deactivate_user(self, user_id: int):
             cur = self.conn.cursor()
             cur.execute("UPDATE users SET active = 0 WHERE id = %s", (user_id,))
             self.conn.commit()
    +
    +    def delete_user(self, username: str):
    +        cur = self.conn.cursor()
    +        cur.execute(f"DELETE FROM users WHERE name = '{username}'")
    +        self.conn.commit()
    +        print("deleted user", username)
    +
    +    def rename_user(self, old: str, new: str):
    +        pass
    diff --git a/src/utils/validators.py b/src/utils/validators.py
    index 3333333..4444444 100644
    --- a/src/utils/validators.py
    +++ b/src/utils/validators.py
    @@ -1,7 +1,11 @@
     """Validators."""
    +import re
     
     
     def validate_email(email: str) -> bool:
    -    return "@" in email and "." in email.split("@")[-1]
    +    pattern = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
    +    return bool(pattern.match(email))
    +
    +
    +def validate_phone(phone: str) -> bool:
    +    return phone.isdigit() and len(phone) >= 10
    '''
)


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> Path:
    """Materialize the synthetic repo on disk."""
    for rel, content in REPO_FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def sample_diff() -> str:
    return SAMPLE_DIFF

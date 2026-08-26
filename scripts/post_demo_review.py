"""Post the demo review to GitHub PR #1."""
import os
import subprocess

env_path = os.path.join(os.environ["LOCALAPPDATA"], "hermes", ".env")
key = ""
with open(env_path, encoding="utf-8") as f:
    for line in f:
        if line.startswith("OPENROUTER_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

env = dict(os.environ)
env["OPENROUTER_API_KEY"] = key
env.pop("PYTHONPATH", None)

# gh CLI is authenticated via keyring; export its token for the GitHub client
gh_tok = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
if gh_tok:
    env["GITHUB_TOKEN"] = gh_tok

proc = subprocess.run(
    [
        r"C:\Users\devil\repolens\.venv\Scripts\python.exe", "-m", "repolens",
        "pr", "saketkumar-18/repolens-demo", "1",
        "--local", r"C:\Users\devil\repolens-demo",
        "--post",
    ],
    env=env, capture_output=True, text=True, timeout=900,
    cwd=r"C:\Users\devil\repolens",
)
print("EXIT:", proc.returncode)
print("--- STDERR ---")
print(proc.stderr[-2500:])
print("--- STDOUT ---")
print(proc.stdout[-1000:])

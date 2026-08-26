"""Run RepoLens against the demo PR with the live LLM (reads key from hermes .env)."""
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

proc = subprocess.run(
    [
        r"C:\Users\devil\repolens\.venv\Scripts\python.exe", "-m", "repolens",
        "pr", "saketkumar-18/repolens-demo", "1",
        "--local", r"C:\Users\devil\repolens-demo",
        "--output", r"C:\Users\devil\repolens\demo-review.md",
    ],
    env=env, capture_output=True, text=True, timeout=600,
    cwd=r"C:\Users\devil\repolens",
)
print("EXIT:", proc.returncode)
print("--- STDERR ---")
print(proc.stderr[-2500:])
print("--- STDOUT head ---")
print(proc.stdout[:800])

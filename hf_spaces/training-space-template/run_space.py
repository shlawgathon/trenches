#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PORT = int(os.environ.get("PORT", "7860"))
LOG_PATH = Path("/tmp/trenches-space.log")
STATUS = {
    "state": "starting",
    "summary": "Initializing training space",
}
LOCK = threading.Lock()


def set_status(state: str, summary: str) -> None:
    with LOCK:
        STATUS["state"] = state
        STATUS["summary"] = summary


def append_log(line: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
        if not line.endswith("\n"):
            fh.write("\n")


def run_and_stream(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    append_log(f"$ {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        append_log(line.rstrip("\n"))
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def upload_output(output_dir: Path) -> None:
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    model_repo = os.environ["MODEL_REPO"]
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message=os.environ.get("UPLOAD_MESSAGE", "Upload Trenches checkpoint"),
    )


def train() -> None:
    entity = os.environ["ENTITY"]
    replay_id = os.environ["REPLAY_ID"]
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-8B")
    git_repo_url = os.environ.get("GIT_REPO_URL", "https://github.com/shlawgathon/trenches.git")
    git_ref = os.environ.get("GIT_REF", "main")
    generation_backend = os.environ.get("GENERATION_BACKEND", "vllm")

    set_status("running", f"Preparing repo for {entity}")
    workroot = Path(tempfile.mkdtemp(prefix="trenches-space-"))
    repo_dir = workroot / "trenches"
    output_dir = workroot / "output"
    cache_dir = workroot / ".cache"
    uv_cache_dir = cache_dir / "uv"
    output_dir.mkdir(parents=True, exist_ok=True)
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["UV_CACHE_DIR"] = str(uv_cache_dir)

    try:
        clone_cmd = ["git", "clone", "--depth", "1"]
        if git_ref:
            clone_cmd.extend(["--branch", git_ref, "--single-branch"])
        clone_cmd.extend([git_repo_url, str(repo_dir)])
        run_and_stream(clone_cmd, env=env)

        python_bin = workroot / ".venv" / "bin" / "python"
        set_status("running", f"Installing training stack for {entity}")
        run_and_stream(["uv", "venv", str(workroot / ".venv"), "--python", "3.12"], env=env)
        run_and_stream(
            ["uv", "pip", "install", "--python", str(python_bin), "-e", "backend[train]", "huggingface_hub"],
            cwd=repo_dir,
            env=env,
        )
        run_and_stream(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_bin),
                "trl==0.29.0",
                "vllm",
            ],
            cwd=repo_dir,
            env=env,
        )

        env["TRL_EXPERIMENTAL_SILENCE"] = "1"
        train_cmd = [
            str(python_bin),
            "-m",
            "trenches_env.training_cli",
            "--model-id",
            model_id,
            "--generation-backend",
            generation_backend,
            "--training-agent",
            entity,
            "--training-stage",
            os.environ.get("TRAINING_STAGE", "stage_1_dense"),
            "--replay-id",
            replay_id,
            "--train-size",
            os.environ.get("TRAIN_SIZE", "4"),
            "--max-steps",
            os.environ.get("MAX_STEPS", "1"),
            "--num-generations",
            os.environ.get("NUM_GENERATIONS", "4"),
            "--per-device-train-batch-size",
            os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"),
            "--gradient-accumulation-steps",
            os.environ.get("GRADIENT_ACCUMULATION_STEPS", "1"),
            "--learning-rate",
            os.environ.get("LEARNING_RATE", "5e-6"),
            "--beta",
            os.environ.get("BETA", "0.001"),
            "--warmup-steps",
            os.environ.get("WARMUP_STEPS", "0"),
            "--temperature",
            os.environ.get("TEMPERATURE", "0.8"),
            "--top-k",
            os.environ.get("TOP_K", "10"),
            "--top-p",
            os.environ.get("TOP_P", "0.95"),
            "--max-prompt-length",
            os.environ.get("MAX_PROMPT_LENGTH", "1024"),
            "--max-completion-length",
            os.environ.get("MAX_COMPLETION_LENGTH", "128"),
            "--save-strategy",
            os.environ.get("SAVE_STRATEGY", "no"),
            "--output-dir",
            str(output_dir),
            "--no-preview",
        ]

        if os.environ.get("QUANTIZE_4BIT", "").lower() in {"1", "true", "yes"}:
            train_cmd.append("--quantize-4bit")

        set_status("running", f"Training {entity}")
        run_and_stream(train_cmd, cwd=repo_dir, env=env)

        set_status("running", f"Uploading checkpoint for {entity}")
        upload_output(output_dir)
        set_status("completed", f"Completed training and upload for {entity}")
    except Exception as exc:
        set_status("failed", f"{type(exc).__name__}: {exc}")
        append_log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        if os.environ.get("KEEP_WORKROOT", "").lower() not in {"1", "true", "yes"}:
            shutil.rmtree(workroot, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        with LOCK:
            state = STATUS["state"]
            summary = STATUS["summary"]
        log_text = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trenches Training Space</title>
  <style>
    body {{ background: #111; color: #eee; font-family: monospace; padding: 24px; }}
    .running {{ color: #ffd166; }}
    .completed {{ color: #06d6a0; }}
    .failed {{ color: #ef476f; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #181818; padding: 16px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Trenches Training Space</h1>
  <p>Status: <span class="{html.escape(state)}">{html.escape(state)}</span></p>
  <p>{html.escape(summary)}</p>
  <pre>{html.escape(log_text[-30000:])}</pre>
</body>
</html>"""
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    thread = threading.Thread(target=train, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

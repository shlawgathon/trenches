from __future__ import annotations

"""
Minimal Colab-oriented TRL training example for the Trenches OpenEnv environment.

Suggested Colab install cell:

    !pip install "trl" "transformers" "datasets" "accelerate" "openenv-core[core]>=0.2.1" "uvicorn"
    !pip install -e /content/trenches/backend

This example is intentionally small:
- launches the backend locally
- connects through the native /openenv route
- runs one GRPO step against the environment reward

It is designed to satisfy the "show a minimal training script" requirement, not to be a full training recipe.
"""

import argparse
import json
import threading
import time
from typing import Any

import httpx
import uvicorn
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from trenches_env.models import AgentAction
from trenches_env.openenv_adapter import TrenchesOpenEnvAction
from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.server import create_app


def _launch_backend(port: int) -> None:
    app = create_app()
    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 30.0
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for backend health at {url}")


def _build_dataset(training_agent: str, size: int) -> Dataset:
    prompt = (
        f"You are the {training_agent} actor in the Trenches crisis environment. "
        "Return strict JSON with keys type, target, and summary. "
        "Only choose one legal action for this actor."
    )
    return Dataset.from_dict({"prompt": [prompt] * size})


def _parse_action(training_agent: str, text: str) -> AgentAction:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}
    return AgentAction(
        actor=training_agent,
        type=payload.get("type", "hold"),
        target=payload.get("target"),
        summary=payload.get("summary", "Fallback hold after invalid JSON completion."),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal TRL/OpenEnv training example for Trenches.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--training-agent", default="us")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--output-dir", default="trl-openenv-minimal")
    parser.add_argument("--train-size", type=int, default=8)
    args = parser.parse_args()

    _launch_backend(args.port)

    client = TrenchesEnvClient(base_url=f"http://127.0.0.1:{args.port}/openenv")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _build_dataset(args.training_agent, args.train_size)

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, Any]:
        rollout = generate_rollout_completions(trainer, prompts)
        completion_ids = rollout["completion_ids"]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        env_rewards: list[float] = []
        for completion in completions:
            client.reset(
                training_agent=args.training_agent,
                training_stage="stage_1_dense",
                max_turns=1,
                include_joint_observations=False,
            )
            observation = client.step(
                TrenchesOpenEnvAction(
                    action=_parse_action(args.training_agent, completion),
                    external_signals=[],
                )
            )
            env_rewards.append(float(observation.reward))

        rollout["env_reward"] = env_rewards
        return rollout

    def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
        rewards = kwargs.get("env_reward")
        if rewards is None:
            return [0.0 for _ in completions]
        return [float(value) for value in rewards]

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=96,
        use_vllm=True,
        vllm_mode="colocate",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=reward_from_env,
        train_dataset=train_dataset,
        rollout_func=rollout_func,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()

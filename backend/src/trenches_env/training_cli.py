from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

import httpx
import uvicorn

from trenches_env.models import AgentAction, Prediction
from trenches_env.openenv_adapter import TrenchesOpenEnvAction, TrenchesOpenEnvObservation
from trenches_env.openenv_client import TrenchesEnvClient
from trenches_env.rl import AGENT_ALLOWED_ACTIONS
from trenches_env.server import create_app

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REPLAY_ID = "us_forecast_seed_2025_2026"
DEFAULT_TRAINING_STAGE = "stage_1_dense"


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


def _build_base_prompt(training_agent: str) -> str:
    return (
        f"You are training the {training_agent} policy in the Trenches OpenEnv historical replay environment. "
        "Return strict JSON only. "
        "Choose one legal action and forecast the next historical event from the visible timeline."
    )


def _render_observation_prompt(
    base_prompt: str,
    training_agent: str,
    observation: TrenchesOpenEnvObservation,
) -> str:
    agent_observation = observation.agent_observation
    public_brief = "\n".join(f"- {item.summary}" for item in agent_observation.public_brief[:4]) or "- None."
    private_brief = "\n".join(f"- {item.summary}" for item in agent_observation.private_brief[:4]) or "- None."
    historical_brief = "\n".join(f"- {line}" for line in agent_observation.historical_brief[:4]) or "- None."
    strategic_state = "\n".join(
        f"- {metric}: {value:.1f}" for metric, value in agent_observation.strategic_state.items()
    ) or "- None."
    available_actions = ", ".join(agent_observation.available_actions)
    return "\n".join(
        [
            base_prompt,
            "",
            f"Training agent: {training_agent}",
            f"Turn: {observation.turn}",
            f"Decision prompt:\n{agent_observation.decision_prompt}",
            "Historical brief:",
            historical_brief,
            "Public brief:",
            public_brief,
            "Private brief:",
            private_brief,
            "Strategic state:",
            strategic_state,
            f"Allowed actions: {available_actions}",
            "Output schema:",
            "{",
            '  "action": {',
            f'    "type": "{agent_observation.available_actions[0] if agent_observation.available_actions else "hold"}",',
            '    "target": "optional_target",',
            '    "summary": "one-sentence action rationale"',
            "  },",
            '  "prediction": {',
            '    "topic": "shipping|border|corridor|domestic|cyber|market|humanitarian|diplomacy|commodities",',
            '    "predicted_actor": "actor or null",',
            '    "predicted_target": "target or null",',
            '    "time_horizon_turns": 1,',
            '    "expected_severity": "low|medium|high|critical",',
            '    "confidence": 0.0,',
            '    "summary": "one-sentence forecast",',
            '    "rationale": "why this next event is likely"',
            "  }",
            "}",
        ]
    )


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def _parse_turn_output(training_agent: str, completion: str) -> tuple[AgentAction, Prediction]:
    payload = _safe_json_loads(completion)
    action_payload = payload.get("action") if isinstance(payload.get("action"), dict) else {}
    prediction_payload = payload.get("prediction") if isinstance(payload.get("prediction"), dict) else {}

    action_type = str(action_payload.get("type") or "hold")
    if action_type not in AGENT_ALLOWED_ACTIONS.get(training_agent, ()):
        action_type = "hold"
    target = action_payload.get("target")
    action_summary = str(
        action_payload.get("summary")
        or "Fallback hold after invalid or partial model completion."
    )

    prediction_topic = str(prediction_payload.get("topic") or "diplomacy")
    predicted_actor = prediction_payload.get("predicted_actor")
    predicted_target = prediction_payload.get("predicted_target")
    prediction_summary = str(
        prediction_payload.get("summary")
        or "Fallback low-confidence forecast after invalid or partial model completion."
    )
    rationale = str(prediction_payload.get("rationale") or "Parser fallback.")
    confidence_raw = prediction_payload.get("confidence", 0.1)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.1

    horizon_raw = prediction_payload.get("time_horizon_turns", 1)
    try:
        time_horizon_turns = max(1, int(horizon_raw))
    except (TypeError, ValueError):
        time_horizon_turns = 1

    expected_severity = str(prediction_payload.get("expected_severity") or "medium")
    if expected_severity not in {"low", "medium", "high", "critical"}:
        expected_severity = "medium"

    return (
        AgentAction(
            actor=training_agent,
            type=action_type,  # type: ignore[arg-type]
            target=target if isinstance(target, str) else None,
            summary=action_summary,
        ),
        Prediction(
            agent_id=training_agent,
            topic=prediction_topic,
            predicted_actor=predicted_actor if isinstance(predicted_actor, str) else None,
            predicted_target=predicted_target if isinstance(predicted_target, str) else None,
            time_horizon_turns=time_horizon_turns,
            expected_severity=expected_severity,  # type: ignore[arg-type]
            confidence=confidence,
            summary=prediction_summary,
            rationale=rationale,
        ),
    )


def _build_dataset(training_agent: str, size: int):
    from datasets import Dataset

    base_prompt = _build_base_prompt(training_agent)
    return Dataset.from_dict({"prompt": [base_prompt] * size})


def _required_training_imports() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, transformers, trl, accelerate, and openenv-core first."
        ) from exc
    try:
        from trl.experimental.openenv import generate_rollout_completions
    except ModuleNotFoundError:
        generate_rollout_completions = None
    return {
        "torch": torch,
        "AutoTokenizer": AutoTokenizer,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "generate_rollout_completions": generate_rollout_completions,
    }


def _resolve_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _can_use_vllm(torch_module: Any) -> bool:
    if not hasattr(torch_module, "cuda") or not torch_module.cuda.is_available():
        return False
    try:
        import vllm  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _generate_rollout_completions_transformers(
    *,
    trainer: Any,
    prompts: list[str],
    tokenizer: Any,
    max_prompt_length: int,
    max_completion_length: int,
) -> dict[str, list[Any]]:
    import torch

    tokenizer.padding_side = "left"
    model = trainer.model
    device = _resolve_model_device(model)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation = model.generate(
        **encoded,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    input_width = encoded["input_ids"].shape[1]

    for batch_index in range(len(prompts)):
        prompt_ids.append(
            encoded["input_ids"][batch_index][encoded["attention_mask"][batch_index].bool()].tolist()
        )
        sample_completion_ids: list[int] = []
        sample_logprobs: list[float] = []

        for step_index, step_scores in enumerate(generation.scores):
            token_position = input_width + step_index
            if token_position >= generation.sequences.shape[1]:
                break
            token_id = int(generation.sequences[batch_index, token_position].item())
            if token_id == tokenizer.pad_token_id:
                break
            sample_completion_ids.append(token_id)
            token_logprob = torch.log_softmax(step_scores[batch_index], dim=-1)[token_id].item()
            sample_logprobs.append(float(token_logprob))
            if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                break

        completion_ids.append(sample_completion_ids)
        logprobs.append(sample_logprobs)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
    }


def _preview_rollouts(
    *,
    model: Any,
    tokenizer: Any,
    training_agent: str,
    port: int,
    replay_id: str,
    training_stage: str,
    samples: int,
    max_completion_length: int,
) -> None:
    import torch

    print("\nPreview rollouts")
    for sample_index in range(samples):
        client = TrenchesEnvClient(base_url=f"http://127.0.0.1:{port}/openenv")
        observation = client.reset(
            training_agent=training_agent,
            training_stage=training_stage,
            max_turns=1,
            replay_id=replay_id,
            episode_id=f"preview-{sample_index}-{int(time.time() * 1000)}",
        )
        prompt = _render_observation_prompt(
            _build_base_prompt(training_agent),
            training_agent,
            observation,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion_ids = output[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        action, prediction = _parse_turn_output(training_agent, completion)
        result = client.step(
            TrenchesOpenEnvAction(
                action=action,
                prediction=prediction,
                external_signals=[],
            )
        )
        actual_event = result.revealed_event.summary if result.revealed_event is not None else "n/a"
        print(
            f"[sample {sample_index + 1}] reward={result.reward:.3f} "
            f"forecast={result.reward_breakdown.forecast_total:.3f} "
            f"action={action.type} topic={prediction.topic} actual={actual_event}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a replay-aware OpenEnv policy for Trenches.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--training-agent", default="us")
    parser.add_argument("--training-stage", default=DEFAULT_TRAINING_STAGE)
    parser.add_argument("--replay-id", default=DEFAULT_REPLAY_ID)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-backend", choices=["auto", "vllm", "transformers"], default="auto")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=220)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="trl-openenv-historical-replay")
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    imports = _required_training_imports()
    torch = imports["torch"]
    AutoTokenizer = imports["AutoTokenizer"]
    GRPOConfig = imports["GRPOConfig"]
    GRPOTrainer = imports["GRPOTrainer"]
    generate_rollout_completions = imports["generate_rollout_completions"]
    generation_backend = args.generation_backend
    if generation_backend == "auto":
        generation_backend = "vllm" if generate_rollout_completions is not None and _can_use_vllm(torch) else "transformers"
    if generation_backend == "vllm" and generate_rollout_completions is None:
        raise RuntimeError("The selected vLLM backend requires `trl.experimental.openenv.generate_rollout_completions`.")

    _launch_backend(args.port)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _build_dataset(args.training_agent, args.train_size)
    base_prompt = _build_base_prompt(args.training_agent)

    def rollout_func(prompts: list[str], trainer: Any) -> dict[str, list[Any]]:
        clients: list[TrenchesEnvClient] = []
        grounded_prompts: list[str] = []

        for index, prompt in enumerate(prompts):
            client = TrenchesEnvClient(base_url=f"http://127.0.0.1:{args.port}/openenv")
            observation = client.reset(
                training_agent=args.training_agent,
                training_stage=args.training_stage,
                max_turns=1,
                replay_id=args.replay_id,
                episode_id=f"train-{index}-{int(time.time() * 1000)}",
            )
            clients.append(client)
            grounded_prompts.append(
                _render_observation_prompt(prompt or base_prompt, args.training_agent, observation)
            )

        if generation_backend == "vllm":
            outputs = generate_rollout_completions(trainer, grounded_prompts)
            rollout_outputs = {
                "prompt_ids": [output["prompt_ids"] for output in outputs],
                "completion_ids": [output["completion_ids"] for output in outputs],
                "logprobs": [output["logprobs"] for output in outputs],
            }
        else:
            rollout_outputs = _generate_rollout_completions_transformers(
                trainer=trainer,
                prompts=grounded_prompts,
                tokenizer=tokenizer,
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
            )
        completion_ids = rollout_outputs["completion_ids"]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        env_rewards: list[float] = []
        forecast_rewards: list[float] = []
        for client, completion in zip(clients, completions, strict=True):
            action, prediction = _parse_turn_output(args.training_agent, completion)
            result = client.step(
                TrenchesOpenEnvAction(
                    action=action,
                    prediction=prediction,
                    external_signals=[],
                )
            )
            env_rewards.append(float(result.reward))
            forecast_rewards.append(float(result.reward_breakdown.forecast_total))

        return {
            "prompt_ids": rollout_outputs["prompt_ids"],
            "completion_ids": completion_ids,
            "logprobs": rollout_outputs["logprobs"],
            "env_reward": env_rewards,
            "forecast_reward": forecast_rewards,
        }

    def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
        rewards = kwargs.get("env_reward")
        if rewards is None:
            return [0.0 for _ in completions]
        return [float(reward) for reward in rewards]

    training_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "num_train_epochs": 1,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "generation_batch_size": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "logging_steps": 1,
        "report_to": [],
        "use_vllm": generation_backend == "vllm",
    }
    if generation_backend == "vllm":
        training_kwargs["vllm_mode"] = "colocate"

    training_args = GRPOConfig(**training_kwargs)

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=reward_from_env,
        train_dataset=train_dataset,
        rollout_func=rollout_func,
        args=training_args,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)

    print("\nTraining complete")
    print(f"Generation backend: {generation_backend}")
    print(json.dumps(train_result.metrics, indent=2, sort_keys=True))

    if not args.no_preview and args.preview_samples > 0:
        _preview_rollouts(
            model=trainer.model,
            tokenizer=tokenizer,
            training_agent=args.training_agent,
            port=args.port,
            replay_id=args.replay_id,
            training_stage=args.training_stage,
            samples=args.preview_samples,
            max_completion_length=args.max_completion_length,
        )


if __name__ == "__main__":
    main()

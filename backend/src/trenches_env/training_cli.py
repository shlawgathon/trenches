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
from trenches_env.training_server import create_training_app

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_REPLAY_ID = "us_synthetic_seed_2025_2026"
DEFAULT_TRAINING_STAGE = "stage_1_dense"
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def _launch_backend(
    port: int,
    *,
    live_source_auto_start: bool = False,
    source_warm_start: bool = False,
) -> None:
    del live_source_auto_start, source_warm_start
    app = create_training_app()
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


def _truncate_prompt_for_model(*, tokenizer: Any, prompt: str, max_prompt_length: int) -> str:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    input_ids = encoded.get("input_ids", [])
    if not input_ids:
        return prompt
    return tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _parse_lora_target_modules(raw_value: str) -> str | list[str]:
    target_modules = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not target_modules:
        raise ValueError("LoRA target modules cannot be empty.")
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        return "all-linear"
    return target_modules


def _preview_rollouts(
    *,
    model: Any,
    tokenizer: Any,
    training_agent: str,
    port: int,
    replay_id: str,
    training_stage: str,
    samples: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> None:
    import torch

    print("\nPreview rollouts")
    for sample_index in range(samples):
        client = TrenchesEnvClient(base_url=f"http://127.0.0.1:{port}/openenv")
        reset_result = client.reset(
            training_agent=training_agent,
            training_stage=training_stage,
            max_turns=1,
            replay_id=replay_id,
            episode_id=f"preview-{sample_index}-{int(time.time() * 1000)}",
        )
        observation = reset_result.observation
        prompt = _render_observation_prompt(
            _build_base_prompt(training_agent),
            training_agent,
            observation,
        )
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 1_000_000:
            model_max_length = max_prompt_length
        preview_prompt_length = min(max_prompt_length, model_max_length)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=preview_prompt_length,
        ).to(model.device)
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
        step_result = client.step(
            TrenchesOpenEnvAction(
                action=action,
                prediction=prediction,
                external_signals=[],
            )
        )
        step_obs = step_result.observation
        actual_event = step_obs.revealed_event.summary if step_obs.revealed_event is not None else "n/a"
        step_reward = step_result.reward if step_result.reward is not None else 0.0
        print(
            f"[sample {sample_index + 1}] reward={step_reward:.3f} "
            f"action={action.type} topic={prediction.topic} actual={actual_event}"
        )


class OpenEnvGRPOTrainer:
    """Force GRPO to use the custom OpenEnv rollout path across generation backends."""

    def _generate_single_turn(self, prompts: list[str]):  # type: ignore[override]
        if getattr(self, "rollout_func", None) is None:
            return super()._generate_single_turn(prompts)

        output = self.rollout_func(prompts, self)
        required_keys = {"prompt_ids", "completion_ids", "logprobs"}
        missing = required_keys.difference(output)
        if missing:
            raise RuntimeError(f"rollout_func is missing required keys: {sorted(missing)}")
        extra_fields = {key: value for key, value in output.items() if key not in required_keys}
        return output["prompt_ids"], output["completion_ids"], output["logprobs"], extra_fields


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
    # Post-training plan args
    parser.add_argument("--quantize-4bit", action="store_true", help="Load model with 4-bit NF4 quantization via bitsandbytes (requires CUDA)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank used with quantized training")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha used with quantized training")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout used with quantized training")
    parser.add_argument(
        "--lora-target-modules",
        default=DEFAULT_LORA_TARGET_MODULES,
        help='Comma-separated LoRA target modules, or "all-linear"',
    )
    parser.add_argument("--beta", type=float, default=0.04, help="KL coefficient for GRPO")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for generation")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--optim",
        default="adamw_torch_fused",
        help="Optimizer passed through to GRPOConfig (for example adamw_bnb_8bit).",
    )
    parser.add_argument("--save-strategy", default="no", choices=["no", "steps", "epoch"], help="Checkpoint save strategy")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps (when save-strategy=steps)")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.0,
        help=(
            "Fraction of total GPU memory reserved for vLLM (0.0 = auto-detect). "
            "vLLM requires total*utilization <= free_memory at init, so this must "
            "account for training + reference model weights already on GPU."
        ),
    )
    parser.add_argument(
        "--vllm-enable-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM sleep mode so training and generation take turns on GPU (default: on).",
    )
    args = parser.parse_args()

    # FORCE-DISABLE RSS FEEDS DURING TRAINING to prevent rate limits and speed up rollouts.
    import os
    os.environ["TRENCHES_DISABLE_RSS"] = "1"

    imports = _required_training_imports()
    torch = imports["torch"]
    AutoTokenizer = imports["AutoTokenizer"]
    GRPOConfig = imports["GRPOConfig"]
    GRPOTrainer = type("OpenEnvGRPOTrainer", (OpenEnvGRPOTrainer, imports["GRPOTrainer"]), {})
    generate_rollout_completions = imports["generate_rollout_completions"]
    generation_backend = args.generation_backend
    if generation_backend == "auto":
        generation_backend = "vllm" if generate_rollout_completions is not None and _can_use_vllm(torch) else "transformers"
    if generation_backend == "vllm" and generate_rollout_completions is None:
        raise RuntimeError("The selected vLLM backend requires `trl.experimental.openenv.generate_rollout_completions`.")

    _launch_backend(
        args.port,
        live_source_auto_start=False,
        source_warm_start=False,
    )

    # Model loading — optionally with 4-bit quantization
    # Always pre-load the model so we can measure GPU memory before GRPOTrainer
    # adds the reference model and vLLM (which need to fit in remaining memory).
    from transformers import AutoModelForCausalLM
    peft_config = None
    if args.quantize_4bit:
        from transformers import BitsAndBytesConfig
        try:
            from peft import LoraConfig, TaskType
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing PEFT dependency. Install backend[train] so quantized training can attach LoRA adapters."
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading {args.model_id} with 4-bit NF4 quantization")
        model_ref = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=_parse_lora_target_modules(args.lora_target_modules),
        )
        print(
            "Attaching LoRA adapters for quantized training "
            f"(r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, "
            f"targets={args.lora_target_modules})"
        )
    else:
        print(f"Loading {args.model_id} in bfloat16")
        model_ref = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Auto-detect vLLM GPU memory utilization.
    # After loading the policy model, measure free memory and estimate how much
    # will remain after the reference model (≈ same size) is loaded by GRPOTrainer.
    # vLLM's init check requires: total_gpu * utilization <= free_memory_at_init.
    if generation_backend == "vllm" and torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / (1024 ** 3)
        total_gib = total_bytes / (1024 ** 3)
        used_gib = total_gib - free_gib
        # GRPOTrainer will clone ~1 more copy (reference model) before vLLM init
        estimated_free_after_ref = free_gib - used_gib  # rough: ref ≈ policy size
        estimated_free_after_ref = max(estimated_free_after_ref, free_gib * 0.5)  # floor

        if args.vllm_gpu_memory_utilization <= 0:
            # Auto-detect: use 90% of estimated post-ref free memory
            auto_util = round((estimated_free_after_ref * 0.90) / total_gib, 2)
            auto_util = max(0.15, min(auto_util, 0.90))
            args.vllm_gpu_memory_utilization = auto_util
            print(
                f"Auto-detected vllm_gpu_memory_utilization={auto_util} "
                f"(GPU: {total_gib:.1f} GiB total, {free_gib:.1f} GiB free after policy, "
                f"~{estimated_free_after_ref:.1f} GiB estimated free after ref model)"
            )
        else:
            max_safe = round(estimated_free_after_ref / total_gib, 2)
            if args.vllm_gpu_memory_utilization > max_safe:
                clamped = max(0.15, max_safe - 0.02)
                print(
                    f"WARNING: vllm_gpu_memory_utilization={args.vllm_gpu_memory_utilization} "
                    f"exceeds estimated safe max {max_safe} — clamping to {clamped} "
                    f"(GPU: {total_gib:.1f} GiB total, {free_gib:.1f} GiB free, "
                    f"~{estimated_free_after_ref:.1f} GiB estimated post-ref)"
                )
                args.vllm_gpu_memory_utilization = clamped

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _build_dataset(args.training_agent, args.train_size)
    base_prompt = _build_base_prompt(args.training_agent)

    def rollout_func(prompts: list[str], trainer: Any) -> dict[str, list[Any]]:
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        logprobs: list[list[float]] = []
        env_rewards: list[float] = []
        forecast_rewards: list[float] = []

        for index, prompt in enumerate(prompts):
            with TrenchesEnvClient(base_url=f"http://127.0.0.1:{args.port}/openenv") as client:
                reset_result = client.reset(
                    training_agent=args.training_agent,
                    training_stage=args.training_stage,
                    max_turns=1,
                    replay_id=args.replay_id,
                    episode_id=f"train-{index}-{int(time.time() * 1000)}",
                )
                grounded_prompt = _render_observation_prompt(
                    prompt or base_prompt,
                    args.training_agent,
                    reset_result.observation,
                )
                if generation_backend == "vllm":
                    rollout_prompt = _truncate_prompt_for_model(
                        tokenizer=tokenizer,
                        prompt=grounded_prompt,
                        max_prompt_length=args.max_prompt_length,
                    )
                    rollout_output = generate_rollout_completions(trainer, [rollout_prompt])[0]
                else:
                    rollout_output = {
                        key: value[0]
                        for key, value in _generate_rollout_completions_transformers(
                            trainer=trainer,
                            prompts=[grounded_prompt],
                            tokenizer=tokenizer,
                            max_prompt_length=args.max_prompt_length,
                            max_completion_length=args.max_completion_length,
                        ).items()
                    }

                completion_text = tokenizer.decode(rollout_output["completion_ids"], skip_special_tokens=True)
                action, prediction = _parse_turn_output(args.training_agent, completion_text)
                step_result = client.step(
                    TrenchesOpenEnvAction(
                        action=action,
                        prediction=prediction,
                        external_signals=[],
                    )
                )

            prompt_ids.append(list(rollout_output["prompt_ids"]))
            completion_ids.append(list(rollout_output["completion_ids"]))
            logprobs.append([float(value) for value in rollout_output["logprobs"]])
            step_reward = step_result.reward if step_result.reward is not None else 0.0
            step_obs = step_result.observation
            forecast_total = step_obs.reward_breakdown.forecast_total if step_obs.reward_breakdown is not None else 0.0
            env_rewards.append(float(step_reward))
            forecast_rewards.append(float(forecast_total))

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
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
        "max_completion_length": args.max_completion_length,
        "logging_steps": 1,
        "report_to": [],
        "use_vllm": generation_backend == "vllm",
        "beta": args.beta,
        "warmup_steps": args.warmup_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "optim": args.optim,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
    }
    if not args.quantize_4bit:
        training_kwargs["bf16"] = True
    if generation_backend == "vllm":
        training_kwargs["vllm_mode"] = "colocate"
        training_kwargs["vllm_max_model_length"] = args.max_prompt_length + args.max_completion_length
        training_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        training_kwargs["vllm_enable_sleep_mode"] = args.vllm_enable_sleep_mode

    training_args = GRPOConfig(**training_kwargs)

    trainer = GRPOTrainer(
        model=model_ref,
        processing_class=tokenizer,
        reward_funcs=reward_from_env,
        train_dataset=train_dataset,
        rollout_func=rollout_func,
        args=training_args,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
        )


if __name__ == "__main__":
    main()

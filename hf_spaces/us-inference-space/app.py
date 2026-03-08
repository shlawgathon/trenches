from __future__ import annotations

import os
import threading
from threading import Thread

os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


MODEL_REPO = os.environ.get("MODEL_REPO", "AlazarM/trenches-us-qwen3-8b-real")
DEFAULT_SYSTEM_PROMPT = """You are the United States strategic actor in Trenches.
Respond like a high-capability policy model: concise, analytical, and grounded in statecraft,
deterrence, coalition management, escalation control, and political tradeoffs."""

_MODEL_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None


def load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _TOKENIZER, _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _TOKENIZER, _MODEL

        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            device_map="auto",
            dtype=torch.float16,
            quantization_config=quantization_config,
        )
        model.eval()
        _TOKENIZER = tokenizer
        _MODEL = model
        return tokenizer, model


def build_messages(message: str, history: list[tuple[str, str]], system_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for user_text, assistant_text in history:
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": message})
    return messages


def respond(
    message: str,
    history: list[tuple[str, str]],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    if not message or not message.strip():
        yield history, history, ""
        return

    tokenizer, model = load_model()
    messages = build_messages(message, history, system_prompt)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded.to(device)
        attention_mask = torch.ones_like(input_ids)
    else:
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "temperature": max(float(temperature), 1e-5),
        "top_p": float(top_p),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    updated_history = history + [(message, "")]
    yield updated_history, updated_history, ""

    partial = ""
    for token in streamer:
        partial += token
        updated_history[-1] = (message, partial)
        yield updated_history, updated_history, ""

with gr.Blocks(title="Trenches US Model") as demo:
    gr.Markdown(
        f"""
        # Trenches US Model
        Chat with the post-trained US checkpoint from `{MODEL_REPO}`.
        The model is loaded in 4-bit mode so it can run on a `t4-small` GPU Space.
        """
    )
    chatbot = gr.Chatbot(label="Dialogue", height=560)
    state = gr.State([])
    message_box = gr.Textbox(label="Message", lines=4, placeholder="Ask the US model for a strategic assessment...")

    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear")

    with gr.Accordion("Generation Settings", open=False):
        system_prompt = gr.Textbox(label="System prompt", value=DEFAULT_SYSTEM_PROMPT, lines=4)
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, value=0.7, step=0.05)
        top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
        max_new_tokens = gr.Slider(label="Max new tokens", minimum=32, maximum=1024, value=256, step=32)

    gr.Examples(
        examples=[
            ["Assess how the US should respond to a sudden Iranian missile launch with allied casualties."],
            ["Draft a short NSC-style recommendation balancing deterrence and escalation risk."],
            ["What coalition move best stabilizes Gulf shipping without triggering direct war?"],
        ],
        inputs=message_box,
    )

    chat_inputs = [message_box, state, system_prompt, temperature, top_p, max_new_tokens]
    chat_outputs = [chatbot, state, message_box]
    message_box.submit(respond, inputs=chat_inputs, outputs=chat_outputs)
    send_button.click(respond, inputs=chat_inputs, outputs=chat_outputs)
    clear_button.click(lambda: ([], [], ""), outputs=[chatbot, state, message_box], queue=False)


if __name__ == "__main__":
    demo.queue(max_size=16).launch(theme=gr.themes.Soft())

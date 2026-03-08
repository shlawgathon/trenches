---
title: Trenches US Model
emoji: 🦅
colorFrom: red
colorTo: blue
sdk: gradio
app_file: app.py
suggested_hardware: t4-small
---

# Trenches US Model

Chat with the post-trained US agent model from Trenches.

- Model: `AlazarM/trenches-us-qwen3-8b-real`
- Runtime: 4-bit `transformers` inference
- Hardware target: `t4-small`

The Space loads the model lazily on first request to keep startup simple and costs lower.

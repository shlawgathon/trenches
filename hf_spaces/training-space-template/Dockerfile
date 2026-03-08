FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRL_EXPERIMENTAL_SILENCE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    PORT=7860

RUN apt-get update \
  && apt-get install -y --no-install-recommends git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY run_space.py /app/run_space.py

EXPOSE 7860

CMD ["python", "/app/run_space.py"]

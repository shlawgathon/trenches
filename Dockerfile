FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    TRENCHES_ENTITIES_ROOT=/app/entities \
    PORT=7860 \
    BACKEND_PROXY_TARGET=http://127.0.0.1:8000 \
    NEXT_PUBLIC_API_BASE_URL=/backend-api \
    NEXT_PUBLIC_VERCEL_API_BASE=/api \
    NEXT_PUBLIC_ENABLE_SOURCE_LOGIC=true

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl ca-certificates unzip \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

COPY package.json bun.lock* ./
RUN bun install --frozen-lockfile

COPY next.config.ts postcss.config.mjs tsconfig.json next-env.d.ts ./
COPY app ./app
COPY src ./src

COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/src ./backend/src
COPY entities ./entities

RUN pip install --no-cache-dir ./backend
RUN bun run build

COPY scripts/start-space.sh ./scripts/start-space.sh
RUN chmod +x ./scripts/start-space.sh

EXPOSE 7860

CMD ["./scripts/start-space.sh"]

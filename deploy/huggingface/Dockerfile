FROM oven/bun:1.2 AS frontend-builder

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY . .

# Assumes the final frontend uses Bun-managed Next.js scripts:
#   "build": "bun --bun next build"
#   "start": "bun --bun next start"
RUN bun install --frozen-lockfile
RUN bun --bun run build


FROM oven/bun:1.2 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRENCHES_ENTITIES_ROOT=/app/entities
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
ENV HOST=127.0.0.1
ENV PORT=8000
ENV FRONTEND_PORT=3000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/src ./backend/src
COPY entities ./entities

WORKDIR /app/backend
RUN python3 -m pip install --no-cache-dir .

WORKDIR /app
COPY --from=frontend-builder /app /app
COPY deploy/huggingface/nginx.conf /etc/nginx/nginx.conf
COPY deploy/huggingface/start.sh /start.sh

RUN chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]

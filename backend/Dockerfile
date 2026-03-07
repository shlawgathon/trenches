FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRENCHES_ENTITIES_ROOT=/app/entities

WORKDIR /app

COPY backend/pyproject.toml backend/README.md ./backend/
COPY backend/src ./backend/src
COPY entities ./entities

WORKDIR /app/backend

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "trenches_env.server"]

# Multi-stage-ish: bake SentenceTransformer so runtime works without HF egress.

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY scripts/prefetch_embedding_model.py scripts/prefetch_embedding_model.py

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Hub during image build (runtime may block huggingface.co). Optional HF_TOKEN for rate limits.
RUN python scripts/prefetch_embedding_model.py

# Prefer local cache after deploy (avoids flaky / blocked HF at startup).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

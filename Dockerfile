# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12.14
ARG UV_VERSION=0.12.19

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /srv

COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN groupadd --gid 10001 limbot \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin limbot

WORKDIR /srv

COPY --from=builder --chown=limbot:limbot /opt/venv /opt/venv
COPY --chown=limbot:limbot app ./app
COPY --chown=limbot:limbot pyproject.toml README.md ./

USER limbot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u, sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"]

# A single worker keeps in-process state consistent: the dedupe cache and the background task
# pool live in this process, and Prometheus metrics cannot be aggregated across workers.
# Scale out with more containers instead, which keeps /metrics scrape-able.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*", \
     "--timeout-keep-alive", "30", \
     "--workers", "1"]

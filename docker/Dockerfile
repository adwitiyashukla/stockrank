FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[explain,market,api,app]"

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="stockrank" \
      org.opencontainers.image.description="Cross-sectional equity return forecasting and portfolio construction" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 stockrank

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --chown=stockrank:stockrank src ./src
COPY --chown=stockrank:stockrank configs ./configs
COPY --chown=stockrank:stockrank dashboard ./dashboard
COPY --chown=stockrank:stockrank scripts ./scripts
COPY --chown=stockrank:stockrank .streamlit ./.streamlit
COPY --chown=stockrank:stockrank pyproject.toml README.md ./

RUN mkdir -p artifacts data/cache reports && chown -R stockrank:stockrank /app
USER stockrank
ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "stockrank.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

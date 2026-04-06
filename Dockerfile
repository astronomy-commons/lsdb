# Multi-stage build for lsdb + Dask runtime image
# Provides a ready-to-use environment for distributed spatial analysis
# of astronomical catalogs with Dask scheduler and worker support.

# === Build stage ===
FROM python:3.12-slim AS builder

ARG LSDB_VERSION=0.0.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /tmp/lsdb

# setuptools_scm requires git metadata to detect the version. Since .git is
# excluded via .dockerignore, we pass the version explicitly. In CI the
# workflow sets this from the release tag; for local builds it defaults to
# 0.0.0 (development).
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${LSDB_VERSION}

RUN pip install --no-cache-dir /tmp/lsdb && \
    pip install --no-cache-dir prometheus-client && \
    rm -rf /tmp/lsdb

# === Runtime stage ===
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/astronomy-commons/lsdb"
LABEL org.opencontainers.image.description="lsdb + Dask runtime for distributed spatial analysis of astronomical catalogs"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -g 1000 lsdb && \
    useradd -u 1000 -g 1000 -m -s /bin/bash lsdb && \
    mkdir -p /data /app && \
    chown -R lsdb:lsdb /data /app

WORKDIR /app

# Dask scheduler (8786) and dashboard (8787)
EXPOSE 8786 8787

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import lsdb; import dask"]

USER lsdb

CMD ["python"]

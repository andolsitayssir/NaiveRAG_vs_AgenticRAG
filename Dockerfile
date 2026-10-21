FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
# BuildKit cache mounts: even if this layer is ever invalidated, wheels and
# HF downloads are reused instead of re-downloaded (BuildKit is default on
# modern Docker / Docker Desktop).
# Install CPU-only torch first: the default PyPI torch pulls gigabytes of
# CUDA libraries that are useless here (no GPU is exposed to this container)
# and makes builds take hours on slow connections.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r /app/requirements.txt

COPY . /app

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8787

ENTRYPOINT ["/app/entrypoint.sh"]

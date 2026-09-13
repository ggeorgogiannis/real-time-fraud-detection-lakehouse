FROM python:3.11-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" fraud \
    && useradd \
        --create-home \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        fraud

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install . \
    && mkdir -p /app/data \
    && chown -R fraud:fraud /app/data

USER fraud

ENTRYPOINT ["fraud-lakehouse"]
CMD ["--help"]
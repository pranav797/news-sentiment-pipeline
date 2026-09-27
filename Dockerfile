# One image for all three services (worker, API, dashboard); docker-compose.yml
# picks the command for each.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# CPU-only torch wheel first (avoids multi-GB CUDA libraries), then the rest of
# the pinned, audited requirements.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install -r requirements.txt

# Unprivileged runtime user. Application code stays root-owned and read-only
# to that user, so a compromised process cannot modify it; only the data and
# model-cache directories are writable.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && mkdir -p /app/data "$HF_HOME" \
 && chown -R app:app /app/data /home/app/.cache

COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/
COPY flows/ flows/
COPY .streamlit/ .streamlit/
COPY scheduler.py .

USER app

CMD ["python", "scheduler.py"]

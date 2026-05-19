# =============================================================================
# F1 Pit Stop — Model Training Container
# =============================================================================
# Build  : docker build -t f1-training .
# Run    : docker run --env-file .env f1-training
# On ECS : credentials come from the ECS Task IAM Role — no .env file needed
# =============================================================================

FROM python:3.11-slim

# Metadata
LABEL project="f1-pitstop" \
      stage="training"

# System dependencies for LightGBM (needs libgomp) and pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
# .env is intentionally NOT copied — credentials are injected at runtime
COPY train.py   .
COPY config.yaml .

# Non-root user for security best practice
RUN useradd --create-home appuser
USER appuser

# Entry point
CMD ["python", "train.py", "--config", "/app/config.yaml"]

FROM python:3.11-slim

WORKDIR /app

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy and install Python dependencies first (better Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Copy application code and generate uv.lock
COPY . .
RUN uv lock

# Expose the port HF Spaces expects
EXPOSE 7860

# Environment defaults for containerized deployment
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
ENV OPENENV_ENV=production
ENV USE_TRANSFORMERS_NER=0
ENV ENABLE_BERT_SCORE=0
ENV INFERENCE_VERBOSE=0
ENV ENABLE_GRADIO_UI=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

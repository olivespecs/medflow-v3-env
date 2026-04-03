FROM python:3.11-slim

WORKDIR /app

ARG PRELOAD_BERTSCORE=0
ARG BERTSCORE_MODEL_TYPE=distilbert-base-uncased
ARG BERTSCORE_METRIC_PATH=bertscore

# Install only the lightweight core stack (no torch/transformers).
# Semantic similarity scoring uses a fast Jaccard fallback automatically.
# For full ML stack locally: pip install -r requirements-ml.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

ENV BERTSCORE_MODEL_TYPE=${BERTSCORE_MODEL_TYPE}
ENV BERTSCORE_METRIC_PATH=${BERTSCORE_METRIC_PATH}

# Optional: pre-fetch BERTScore metric/model into image layers to avoid first-request downloads.
# When PRELOAD_BERTSCORE=1, the ML stack (torch, transformers, evaluate) is installed
# automatically from requirements-ml.txt before running the preload script.
COPY requirements-ml.txt ./
RUN if [ "$PRELOAD_BERTSCORE" = "1" ]; then \
      pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-ml.txt && \
      python -c "import os, evaluate; metric = evaluate.load(os.getenv('BERTSCORE_METRIC_PATH', 'bertscore')); metric.compute(predictions=['warmup'], references=['warmup'], lang='en', model_type=os.getenv('BERTSCORE_MODEL_TYPE', 'distilbert-base-uncased')); print('BERTScore artifacts preloaded')"; \
    fi

COPY . .

EXPOSE 7860

# OPENAI_API_KEY is optional -- /baseline and --demo work without it.
ENV OPENAI_API_KEY=""

# Run the app explicitly as src.main:app so Hugging Face Spaces ingress routes it properly
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]

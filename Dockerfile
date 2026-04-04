FROM python:3.11-slim

WORKDIR /app

ARG PRELOAD_BERTSCORE=0
ARG BERTSCORE_MODEL_TYPE=distilbert-base-uncased
ARG BERTSCORE_METRIC_PATH=bertscore

# Install all dependencies (core + ML stack + testing).
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

ENV BERTSCORE_MODEL_TYPE=${BERTSCORE_MODEL_TYPE}
ENV BERTSCORE_METRIC_PATH=${BERTSCORE_METRIC_PATH}

# Optional: pre-fetch BERTScore metric/model into image layers to avoid first-request downloads.
RUN if [ "$PRELOAD_BERTSCORE" = "1" ]; then \
      python -c "import os, evaluate; metric = evaluate.load(os.getenv('BERTSCORE_METRIC_PATH', 'bertscore')); metric.compute(predictions=['warmup'], references=['warmup'], lang='en', model_type=os.getenv('BERTSCORE_MODEL_TYPE', 'distilbert-base-uncased')); print('BERTScore artifacts preloaded')"; \
    fi

COPY . .

EXPOSE 7860

# OPENAI_API_KEY is optional -- /baseline and --demo work without it.
ENV OPENAI_API_KEY=""
ENV USE_TRANSFORMERS_NER=1

# Run the app explicitly as src.main:app so Hugging Face Spaces ingress routes it properly
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]

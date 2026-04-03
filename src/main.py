"""Uvicorn entry point — mounts Gradio UI at /ui, serves FastAPI API at root."""

# Load .env file before anything else so env vars are available everywhere.
# Silently ignored if python-dotenv is not installed or .env doesn't exist.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import logging
from contextlib import asynccontextmanager
import gradio as gr
import uvicorn
from fastapi import FastAPI

from .api import app as fastapi_app
from .ui import create_ui
from .mcp_server import mcp

logger = logging.getLogger(__name__)


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    Lifespan context manager for the main app.
    Handles startup (model warmup) and shutdown events.
    """
    # Startup: Warm up ML models
    import sys
    
    # Check if we're in a high-resource environment
    is_colab = "COLAB_GPU" in os.environ or "google.colab" in sys.modules
    
    # Warm up NER model if enabled
    use_ner = os.getenv("USE_TRANSFORMERS_NER", "1" if is_colab else "0")
    if use_ner == "1":
        try:
            logger.info("Warming up NER model")
            from .baseline_agent import get_ner_agent
            agent = get_ner_agent()
            if agent.nlp:
                # Warm up with dummy text
                agent.redact_text("test patient John Doe MRN123456")
                logger.info("NER model warmed up successfully")
            else:
                logger.info("NER model not loaded (disabled)")
        except Exception as e:
            logger.warning("Failed to warm up NER model: %s", e)
    
    # Warm up BERTScore if enabled
    enable_bert = os.getenv("ENABLE_BERT_SCORE", "1" if is_colab else "0")
    if enable_bert == "1":
        try:
            logger.info("Warming up BERTScore model")
            from .utils import semantic_similarity_score
            # Trigger model loading with a dummy comparison
            semantic_similarity_score("test reference text", "test candidate text")
            logger.info("BERTScore model warmed up successfully")
        except Exception as e:
            logger.warning("Failed to warm up BERTScore model: %s", e)
    
    if use_ner != "1" and enable_bert != "1":
        logger.info("No ML models to warm up (all disabled for fast startup)")
    
    yield  # Application runs here
    
    # Shutdown: cleanup if needed
    logger.info("Shutting down gracefully")


# Mount Gradio and MCP at import time so Gradio startup hooks (queue workers)
# initialize correctly before handling requests.
app = fastapi_app
original_router_lifespan = app.router.lifespan_context
gradio_ui = create_ui()
gr.mount_gradio_app(app, gradio_ui, path="/")
try:
    app.mount("/mcp", mcp.http_app())
except Exception as e:
    logger.warning("Could not mount MCP server: %s", e)

@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combine API lifespan (from api.py) with app lifespan (warmup)."""
    # Run API lifespan first (episode cleanup task)
    async with original_router_lifespan(app):
        # Then run warmup lifespan
        async with app_lifespan(app):
            yield

app.router.lifespan_context = combined_lifespan


def main() -> None:
    """Console script entrypoint used by OpenEnv validators."""
    uvicorn.run("src.main:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

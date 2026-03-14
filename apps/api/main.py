from fastapi import FastAPI
from src.config.logging import configure_logging
from src.config.settings import settings

configure_logging()

app = FastAPI(
    title="LLM Reliability Lab API",
    version="0.1.0",
    description="API for RAG reliability evaluation runs and reports.",
)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "project": "llm-reliability-lab",
    }
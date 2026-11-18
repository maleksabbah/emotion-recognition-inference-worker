"""Root entry point — runs the worker."""
import asyncio
import logging

from app.Config import MODEL_PATH
from app.ModelFetch import ensure_model
from app.main import run

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_model(MODEL_PATH)
    asyncio.run(run())

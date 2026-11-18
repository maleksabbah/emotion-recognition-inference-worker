"""
Fetch the model checkpoint from S3/R2 at startup.

If MODEL_S3_KEY is set and MODEL_PATH does not exist yet, the checkpoint is
downloaded from S3_BUCKET/MODEL_S3_KEY to MODEL_PATH. A failed download is
fatal: otherwise Predictor would silently fall back to random weights.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def ensure_model(model_path: str) -> None:
    key = os.getenv("MODEL_S3_KEY", "")
    if not key or os.path.exists(model_path):
        return

    import boto3

    bucket = os.environ["S3_BUCKET"]
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_INTERNAL_ENDPOINT") or None,
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.getenv("S3_REGION", "auto"),
    )
    tmp = model_path + ".part"
    logger.info("Downloading model s3://%s/%s -> %s", bucket, key, model_path)
    s3.download_file(bucket, key, tmp)
    os.replace(tmp, model_path)
    logger.info("Model downloaded (%d bytes)", os.path.getsize(model_path))

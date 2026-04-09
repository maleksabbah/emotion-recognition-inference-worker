# EmotionRecognitionInferenceWorker (GPU)
# Stateless Kafka consumer: loads MultiStreamEmotionNet, runs emotion inference
# Talks to: Kafka only (inference_tasks → inference_results)
# Model: mounted as volume at /app/models/best_model.pth
# No HTTP port — scale with: docker compose up --scale inference-worker=2
#
# NOTE: If you don't have an NVIDIA GPU, use Dockerfile.cpu instead

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

CMD ["python", "main.py"]

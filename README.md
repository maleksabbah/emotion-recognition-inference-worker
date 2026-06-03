# EmotionRecognitionInference

Model inference worker for the Mntis platform. Consumes
`InferenceTask` messages, runs the multi-stream emotion CNN on the
crops, and publishes an `InferenceResult` with the top emotion,
full per-class probabilities, valence, arousal, and intensity.

Built as a long-running Kafka consumer (no HTTP surface). Uses
PyTorch + PIL for preprocessing.

## Architecture

The service follows a clean, layered architecture where each layer has
one responsibility and depends only on the layer beneath it:

* **Main loop** — `main.py` runs a Kafka consumer on
  `inference_tasks` and an aiokafka producer for `inference_results`.
  Loads the model once at startup, holds it in memory for the life
  of the process. Replaces what would otherwise be HTTP routes.
* **Services** — the business logic. `InferenceService` unpacks
  the typed `InferenceTask` (base64 crops) into the dict the
  predictor expects, calls the predictor, and wraps the output as
  a typed `InferenceResult`. `Predictor` owns the model — decodes
  each crop with PIL, applies the same `Resize → ToTensor →
  Normalize(ImageNet)` transforms training used, runs the network,
  and turns each of the four heads (emotion / valence / arousal /
  intensity) into `{label, confidence, probabilities}`. Knows
  nothing about Kafka.
* **Model** — `MultiStreamEmotionNet`: a ResNet18 face encoder
  (224×224) + four region encoders (64×64 each for eyes, mouth,
  cheek, forehead) + feature-attention fusion + a shared FC trunk
  + four classification heads. Loaded from
  `/models/multi_stream_emotion.pt` at startup. The architecture
  mirrors the V2 training checkpoint exactly.
* **Repositories** — data access. None for the data path
  (everything is in-memory tensors); Kafka producer / consumer
  wrappers live in `main.py`. The service layer never touches
  raw Kafka directly.
* **Dtos** — `InferenceTask` (input — session_id, frame_number,
  face_index, face + four region crops as base64) and
  `InferenceResult` (output — emotion scores, top emotion +
  confidence, valence, arousal, intensity).
* **Config** — wiring: Kafka consumer (group
  `inference-workers`, reads `inference_tasks`, writes
  `inference_results`), Kafka producer, model path, device
  (auto-selects CUDA if available), label lists for each head.

This separation keeps the queue layer swappable, the business logic
testable in isolation, and the model free to evolve without touching
the rest.

A few things worth calling out: preprocessing is PIL all the way
down (no cv2) because cv2's BGR/RGB roundtrip and `cv2.resize`
interpolation defaults gave subtly different pixel values than
training's PIL pipeline — enough to flip predictions on hard cases
even when the model loaded correctly. The label lists are derived
from `Face.EMOTIONS / VALENCES / AROUSALS` in the training repo to
keep prod and training in sync; if you reorder one, you must
reorder the other.

Part of a multi-service system — see the [platform overview](../EmotionRecognitionDocker)
for the full architecture, pipeline flow, and the other services.

"""
Inference services.

  Model              the PyTorch network                ← existing
  Predictor          loads weights, runs inference      ← existing
  InferenceService   task in → result out, async wrapper
"""
from app.Services.InferenceService import InferenceService

__all__ = ["InferenceService"]
"""
Inference worker repositories — transport boundaries.

  KafkaConsumer    consume inference_tasks
  KafkaProducer    publish inference_results
  RedisRepository  dequeue live tasks
"""
from app.Repositories.KafkaConsumer import KafkaConsumer
from app.Repositories.KafkaProducer import KafkaProducer
from app.Repositories.RedisRepository import RedisRepository

__all__ = ["KafkaConsumer", "KafkaProducer", "RedisRepository"]
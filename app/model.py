import os
import logging

import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/model-cache")


class TranscriptionModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.is_loaded = False

    def load(self):
        logger.info(f"Loading processor from {MODEL_ID}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        logger.info(f"Loading model from {MODEL_ID} in bfloat16...")
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )

        self.is_loaded = True
        logger.info("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        audio = Audio.from_file(audio_path, strict=False)
        audio.resample(self.processor.feature_extractor.sampling_rate)
        duration = len(audio.audio_array) / self.processor.feature_extractor.sampling_rate

        inputs = self.processor(audio.audio_array, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        logger.info(f"Transcribed {duration:.1f}s audio in {len(text)} chars")
        return text, duration

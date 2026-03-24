import os
import logging

import torch
import librosa
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/model-cache")


class TranscriptionModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.is_loaded = False

    def load(self):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        logger.info(f"Loading processor from {MODEL_ID}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

        logger.info(f"Loading model from {MODEL_ID} with 8-bit quantization...")
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )

        self.is_loaded = True
        logger.info("Model loaded successfully.")

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audio_array, sr = librosa.load(audio_path, sr=sampling_rate)
        duration = len(audio_array) / sampling_rate

        inputs = self.processor(audio_array, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return text, duration

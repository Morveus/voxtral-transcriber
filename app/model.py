import os
import math
import logging
import threading

import numpy as np
import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from mistral_common.tokens.tokenizers.audio import Audio

logger = logging.getLogger(__name__)

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/model-cache")

# Le modèle Realtime produit un jeton par trame audio (80 ms) : sans borne,
# transformers génère autant de jetons que l'audio en contient (21243 pour
# 28 minutes), avec un cache qui grossit à chaque pas. On découpe donc l'audio
# en tranches et on plafonne chaque génération.
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "60"))
# Fenêtre, en fin de tranche, où l'on cherche le passage le plus calme pour
# couper entre deux mots plutôt qu'au milieu d'un.
CUT_SEARCH_SECONDS = float(os.environ.get("CUT_SEARCH_SECONDS", "5"))
# Marge de jetons au delà de la durée de la tranche (délai du modèle, fin).
TOKEN_MARGIN = int(os.environ.get("TOKEN_MARGIN", "32"))
# Borne haute absolue par génération, quelle que soit la tranche.
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "2048"))


def cut_points(samples: np.ndarray, sr: int, chunk_s: float, search_s: float) -> list[int]:
    """Bornes des tranches : environ `chunk_s` secondes chacune, coupées au
    point le plus silencieux des `search_s` dernières secondes."""
    n = len(samples)
    size = int(chunk_s * sr)
    if size <= 0 or n <= size:
        return [0, n]
    frame = max(1, int(0.1 * sr))
    bounds = [0]
    start = 0
    while n - start > size:
        end = start + size
        lo = max(start + frame, end - int(search_s * sr))
        best, best_energy = end, None
        for pos in range(lo, end - frame + 1, frame):
            seg = samples[pos:pos + frame]
            energy = float(np.mean(seg.astype(np.float32) ** 2))
            if best_energy is None or energy < best_energy:
                best, best_energy = pos + frame // 2, energy
        bounds.append(best)
        start = best
    bounds.append(n)
    return bounds


class TranscriptionModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.is_loaded = False
        # Une seule carte : une transcription à la fois.
        self.lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.lock.locked()

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

    def tokens_per_second(self) -> float:
        fe = self.processor.feature_extractor
        hop = getattr(fe, "hop_length", 160)
        per_tok = getattr(self.model.config, "audio_length_per_tok", 8)
        return fe.sampling_rate / (hop * per_tok)

    def max_new_tokens(self, seconds: float) -> int:
        wanted = math.ceil(seconds * self.tokens_per_second()) + TOKEN_MARGIN
        return max(TOKEN_MARGIN, min(wanted, MAX_NEW_TOKENS_CAP))

    def _generate(self, samples: np.ndarray, seconds: float) -> str:
        inputs = self.processor(samples, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens(seconds))
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        audio = Audio.from_file(audio_path, strict=False)
        sr = self.processor.feature_extractor.sampling_rate
        audio.resample(sr)
        samples = audio.audio_array
        duration = len(samples) / sr

        bounds = cut_points(samples, sr, CHUNK_SECONDS, CUT_SEARCH_SECONDS)
        parts = []
        with self.lock:
            for i in range(len(bounds) - 1):
                chunk = samples[bounds[i]:bounds[i + 1]]
                seconds = len(chunk) / sr
                text = self._generate(chunk, seconds)
                if text:
                    parts.append(text)
                if len(bounds) > 2:
                    logger.info(f"Chunk {i + 1}/{len(bounds) - 1} ({seconds:.1f}s): {len(text)} chars")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        text = " ".join(parts)
        logger.info(f"Transcribed {duration:.1f}s audio in {len(bounds) - 1} chunk(s), {len(text)} chars")
        return text, duration

# Voxtral Transcriber

A self-hosted speech-to-text API service using [Mistral's Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) model, deployed on Kubernetes with GPU acceleration.

## Features

- REST API endpoint for audio transcription
- Supports multiple audio formats (mp3, wav, ogg, flac, m4a, webm)
- Multilingual transcription (13 languages including French, English, etc.)
- 8-bit quantization via bitsandbytes — runs on GPUs with 12GB+ VRAM
- Kubernetes-ready with NVIDIA GPU support

## API

### POST /transcribe

Upload an audio file and receive its transcription.

**Request:** `multipart/form-data` with an `audio` file field.

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@recording.ogg"
```

**Response:**
```json
{
  "text": "Bonjour, comment ça va ?",
  "duration_seconds": 2.5,
  "processing_time_seconds": 0.8
}
```

### GET /health

Health check endpoint.

## Deployment

### Docker

```bash
docker build -t voxtral-transcriber .
docker run --gpus all -p 8000:8000 voxtral-transcriber
```

The model is downloaded from HuggingFace on first startup. Mount a volume to `/app/model-cache` to persist the model between restarts.

### Kubernetes

Example K8s manifests can be adapted from the Docker setup. The deployment expects:
- A node with GPU access
- NVIDIA GPU runtime class configured

## Architecture

- **Runtime:** Python 3.11 + FastAPI + Uvicorn
- **Model:** Voxtral Mini 4B Realtime (BF16 → 8-bit via bitsandbytes)
- **GPU:** NVIDIA with CUDA, ~4GB VRAM usage in 8-bit mode
- **Storage:** Longhorn PVC for model cache (avoids re-download on restart)

import io
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

PREDICTIONS = Counter("predictions_total", "Predictions by label", ["label"])
ERRORS = Counter("prediction_errors_total", "Rejected prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Inference latency")
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Top-class probability",
    buckets=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)
MAX_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.model import load_classifier  # imported here so tests don't need torch

    app.state.predictor = load_classifier()
    yield


app = FastAPI(
    title="Lesion classifier API (learning project, not for clinical use)",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    p = getattr(request.app.state, "predictor", None)
    return {
        "status": "ok" if p else "starting",
        "model": "trained" if getattr(p, "trained", False) else "untrained-demo",
    }


@app.post("/predict")
async def predict(request: Request, file: Annotated[UploadFile, File()]):
    data = await file.read()
    if len(data) > MAX_BYTES:
        ERRORS.inc()
        raise HTTPException(413, "File too large (max 10 MB)")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        ERRORS.inc()
        raise HTTPException(400, "Uploaded file is not a valid image") from None

    start = time.perf_counter()
    result = request.app.state.predictor.predict(image)
    LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS.labels(label=result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

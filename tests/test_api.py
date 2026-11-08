import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


class FakePredictor:
    trained = True

    def predict(self, image):
        return {"label": "nv", "confidence": 0.91, "probabilities": {"nv": 0.91, "mel": 0.09}}


# No `with` block -> lifespan (model loading) is skipped; we inject a fake.
app.state.predictor = FakePredictor()
client = TestClient(app)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_image():
    r = client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "nv"
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image():
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_metrics_exposed_after_prediction():
    client.post("/predict", files={"file": ("x.png", _png(), "image/png")})
    text = client.get("/metrics").text
    assert 'predictions_total{label="nv"}' in text
    assert "prediction_latency_seconds" in text

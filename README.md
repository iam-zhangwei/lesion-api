# Lesion classifier API — a small end-to-end ML deployment project

A learning project that takes an image classifier (ResNet18, skin-lesion classes from HAM10000)
from training to a containerised, tested and monitored service.

> **Not a medical device. For learning only. Do not use for clinical decisions.**

## What it demonstrates
- **Serving:** FastAPI app with `POST /predict`, `GET /health`, `GET /metrics`
- **Tracking:** training script logs params, per-epoch metrics and the model to MLflow
- **Testing + CI:** pytest API tests; GitHub Actions runs lint, tests and a Docker build on every push
- **Containerisation:** slim Python image, CPU-only PyTorch, non-root user, healthcheck
- **Monitoring:** Prometheus metrics (request counts per label, latency, prediction confidence);
  optional Prometheus + Grafana via Docker Compose

![CI](https://github.com/holyleaf/lesion-api/actions/workflows/ci.yml/badge.svg)

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload
```
Without a weights file the API serves an **untrained demo model** (`/health` reports
`"model": "untrained-demo"`). Put trained weights at `models/model.pt` to use a real model.

```bash
curl -F "file=@some_image.jpg" http://localhost:8000/predict
```

## Train (CPU-friendly)
Arrange images as `data/train/<class>/*.jpg` and `data/val/<class>/*.jpg`, then:
```bash
pip install -r requirements-train.txt
python training/train.py --data-dir data --epochs 3 --subset 2000 --freeze-backbone
mlflow ui          # compare runs at http://localhost:5000
```
Data is not included (HAM10000 is available from its original sources).

## Run with Docker
```bash
docker build -t lesion-api .
docker run -p 8000:8000 -v $(pwd)/models:/srv/models:ro lesion-api
# or the full stack with Prometheus (9090) and Grafana (3000):
docker compose up --build
```

## Verification and monitoring
- API predictions were checked against the training notebook on 9 validation images
  (same classes, same confidences), both in a local run and inside the Docker container.
- Prometheus scrapes the API's `/metrics` endpoint (Docker Compose):

![Prometheus target](docs/prometheus-targets.png)
![Predictions by class](docs/prometheus-predictions.png)



## Limitations / next steps
- Class imbalance is not handled in this training script (a weighted loss or sampler would be the next step).
- Monitoring covers service and confidence metrics; there is no data-drift detection yet.
- No authentication, model registry or deployment to a cluster yet.

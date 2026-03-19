# MLOps Lab 1 - Sentiment Analysis API

FastAPI application for text sentiment analysis using Sentence Transformer embeddings and logistic regression.

## Setup
```bash
uv sync
```

## Run
```bash
uv run uvicorn app:app --reload
```

## Run with Docker
```bash
docker compose up --build
```

## Test
```bash
uv run pytest tests/ -v
```

## API

**POST** `/predict`

Request:
```json
{"text": "What a great MLOps lecture"}
```

Response:
```json
{"prediction": "positive"}
```

Possible values: `negative`, `neutral`, `positive`
"""FastAPI application for sentiment analysis inference."""

from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Mapping from model's numeric output to human-readable sentiment labels
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Paths to serialized model files
TRANSFORMER_PATH = "model/sentence_transformer.model"
CLASSIFIER_PATH = "model/classifier.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models once at application startup."""
    app.state.transformer = SentenceTransformer(TRANSFORMER_PATH)
    app.state.classifier = joblib.load(CLASSIFIER_PATH)
    yield


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Request model containing text to analyze."""

    text: str = Field(..., min_length=1, description="Text to classify sentiment for")


class PredictionResponse(BaseModel):
    """Response model containing predicted sentiment label."""

    prediction: str = Field(..., description="Predicted sentiment: negative, neutral, or positive")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict sentiment of the given text.

    Pipeline: text -> sentence transformer embedding -> logistic regression -> sentiment label
    """
    embedding = app.state.transformer.encode([request.text])
    prediction = app.state.classifier.predict(embedding)
    label = LABEL_MAP[int(prediction[0])]
    return PredictionResponse(prediction=label)

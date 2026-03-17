from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transformer = SentenceTransformer("model/sentence_transformer.model")
    app.state.classifier = joblib.load("model/classifier.joblib")
    yield


app = FastAPI(lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    embedding = app.state.transformer.encode([request.text])
    prediction = app.state.classifier.predict(embedding)
    label = LABEL_MAP[int(prediction[0])]
    return PredictionResponse(prediction=label)

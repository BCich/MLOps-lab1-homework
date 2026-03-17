from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(prediction="positive")

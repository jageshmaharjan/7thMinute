from typing import Dict, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_sentiment_model
#from .entity_recognition.model import Model, get_er_model

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class EntityRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float


class EntityRecognitionResponse(BaseModel):
    tokens: str
    entity: str


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest, model: Model = Depends(get_sentiment_model)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )


@app.post("/entity_recognition", responses=Any)
async def entity_recognition(request: EntityRequest):  #, model: Model = Depends(get_er_model)
    return "Entity Recognition Coming soon..."
#     tokens, entity = model.predict(request.text)
#     return EntityRecognitionResponse(
#         tokens= tokens, entity = entity
#     )


@app.get("/")
async def main():
    return {"Greet": "Sentiment Analyzer!"}

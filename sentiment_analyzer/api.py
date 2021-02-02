from typing import Dict, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_sentiment_model
from .entity_recognition.model import Model, get_er_model
from .entity_mapper import mapping_compute

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


class EntityRequest(BaseModel):
    text: str


class EntityMappingRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float


class EntityRecognitionResponse(BaseModel):
    tokens: str
    entity: str


class EntityMappingResponse(BaseModel):
    tokens: str
    entity: str


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest, model: Model = Depends(get_sentiment_model)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )


@app.post("/entity_recognition", response_model=EntityRecognitionResponse)
async def entity_recognition(request: EntityRequest, model: Model = Depends(get_er_model)):  #, model: Model = Depends(get_er_model)
    tokens, entity = model.predict(request.text)
    return EntityRecognitionResponse(
        tokens= str(tokens), entity = str(entity)
    )


@app.post("/entity_mapping", response_model=EntityMappingResponse)
async def entity_mapping(request: EntityMappingRequest):
    tokens, entity = mapping_compute.compute(request.text)
    return EntityMappingResponse(
        tokens=str(tokens),
        entity=str(entity)
    )


@app.get("/")
async def main():
    return {"Delvify_v.11": "Sentiment Analyzer!", "Delvify_v.12":"Entity Recognition!"}

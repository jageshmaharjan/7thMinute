import os
from secrets import token_urlsafe
from typing import Dict, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from pydantic.schema import timedelta
from starlette.middleware.cors import CORSMiddleware

from editor.config_editor import get_config, view_models
from .classifier.model import Model, get_sentiment_model
from .entity_recognition.model import Model, get_er_model
from .entity_mapper import mapping_compute
from utils import deps
from dotenv import load_dotenv
from pathlib import Path


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


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


@app.post("/token", response_model=deps.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = deps.authenticate_user(deps.users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=deps.status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=deps.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = deps.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=deps.User)
async def read_users_me(current_user: deps.User = Depends(deps.get_current_active_user)):
    return current_user


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: SentimentRequest, model: Model = Depends(get_sentiment_model),
                    current_user: deps.User = Depends(deps.get_current_active_user)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )


@app.post("/entity_recognition", response_model=EntityRecognitionResponse)
async def entity_recognition(request: EntityRequest, model: Model = Depends(get_er_model),
                             current_user: deps.User = Depends(deps.get_current_active_user)):  #, model: Model = Depends(get_er_model)
    tokens, entity = model.predict(request.text)
    return EntityRecognitionResponse(
        tokens= str(tokens), entity = str(entity)
    )


@app.post("/entity_mapping", response_model=EntityMappingResponse)
async def entity_mapping(request: EntityMappingRequest,
                         current_user: deps.User = Depends(deps.get_current_active_user)):
    tokens, entity = mapping_compute.compute(request.text)
    return EntityMappingResponse(
        tokens=str(tokens),
        entity=str(entity)
    )


@app.post('/er_models')
async def select_er_models(current_user: deps.User = Depends(deps.get_current_active_user)):
    # if current_user
    models = view_models()
    return str(models)


@app.post('/sentiment_models')
async def select_sentiment_models(current_user: deps.User = Depends(deps.get_current_active_user)):
    models = view_models()
    return str(models)


@app.post('/config')
async def config_editor(current_user: deps.User = Depends(deps.get_current_active_user)):
    config = get_config()
    models = view_models()
    return config, models


@app.get("/")
async def main():
    return {"Delvify_v.11": "Sentiment Analyzer!", "Delvify_v.12":"Entity Recognition!"}

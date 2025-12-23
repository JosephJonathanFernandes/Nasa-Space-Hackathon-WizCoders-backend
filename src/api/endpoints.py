"""
API endpoints for Exoscope backend (modular, secure, production-ready).
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import traceback
import os
import io
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src.services.model_service import ModelService
from src.services.rag_service import RAGService
from src.services.lightcurve_service import LightCurveService

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    retrieved: List[str]

class PredictRequest(BaseModel):
    features: Optional[List[float]] = None

class ReindexRequest(BaseModel):
    pdf_path: str

# --- Services ---
model_service = ModelService()
rag_service = RAGService()
lightcurve_service = LightCurveService()

# --- Endpoints ---
@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    return await model_service.predict(file)

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await rag_service.chat(req)

@router.get("/stream_chat")
async def stream_chat(question: str):
    return await rag_service.stream_chat(question)

@router.post("/reindex")
async def reindex(req: ReindexRequest):
    return await rag_service.reindex(req)

@router.get("/lightcurve/{kepler_id}")
async def get_lightcurve(kepler_id: str, mission: str = "Kepler"):
    return await lightcurve_service.get_lightcurve(kepler_id, mission)

@router.post("/upload_fits")
async def upload_fits_file(file: UploadFile = File(...)):
    return await lightcurve_service.upload_fits_file(file)

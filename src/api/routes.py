"""
Route registration for Exoscope backend.
"""
from fastapi import FastAPI
from src.api import endpoints

def register_routes(app: FastAPI):
    app.include_router(endpoints.router)

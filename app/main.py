# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Import configuration and routers
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.services.query_service import QueryService

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Efficient, explainable LLM-powered document query system with RAG capabilities."
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1", tags=["queries"])

# Initialize your core query service
query_service = QueryService()



@app.get("/")
async def root():
    return {"message": "Welcome! Use POST /api/v1/hackrx/run to submit queries."}


@app.get("/health")
def healthcheck():
    return {"status": "ok"}

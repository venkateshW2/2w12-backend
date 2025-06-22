# main.py (NEW MODULAR VERSION)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_settings
from api import health, audio
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="2W12.ONE Audio Analysis Platform",
    description="Professional audio analysis with modular architecture",
    version="3.0.0"
)

# Load settings
settings = get_settings()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router)
app.include_router(audio.router, prefix="/api/audio", tags=["Audio Analysis"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "2W12.ONE Audio Analysis Platform",
        "version": "3.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info"
    )
# Imports first
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import health, audio
from api.streaming import router as streaming_router  # This import works now
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="2W12.ONE Audio Analysis Platform",
    description="Professional audio analysis with modular architecture",
    version="3.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers AFTER app is created
app.include_router(health.router)
app.include_router(audio.router, prefix="/api/audio", tags=["Audio Analysis"])

# Include visualization router
try:
    from api.visualization import router as visualization_router
    app.include_router(visualization_router)
    logger.info("✅ Visualization router imported successfully!")
except Exception as e:
    logger.error(f"❌ Failed to import visualization router: {e}")

# Include streaming router
app.include_router(streaming_router)  # This should work now
logger.info("✅ Streaming router included successfully!")

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
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
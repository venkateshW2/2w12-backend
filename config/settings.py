# config/settings.py
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from typing import List
import os

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "https://2w12.one",
        "https://www.2w12.one",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://128.199.25.218:8001"
    ]
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    TEMP_DIR: str = "/tmp/2w12-temp"
    
    # Audio processing
    DEFAULT_SAMPLE_RATE: int = 44100  # Updated to higher quality
    
    class Config:
        env_file = ".env"

# Singleton settings instance
_settings = None

def get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
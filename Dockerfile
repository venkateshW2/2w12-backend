# Use Python 3.10-slim for better compatibility
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    wget \
    build-essential \
    pkg-config \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install requirements with error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Some packages failed, trying without problematic ones..." && \
     pip install --no-cache-dir -r requirements.txt --force-reinstall --no-deps || \
     pip install --no-cache-dir $(grep -v "acoustid\|essentia" requirements.txt))

# Install PyTorch (try GPU first, fallback to CPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
    pip install --no-cache-dir torch torchvision torchaudio

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads temp logs static

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Run command
CMD ["python", "main.py"]
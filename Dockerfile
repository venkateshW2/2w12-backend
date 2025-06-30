# Use TensorFlow GPU base for Week 2 ML capabilities
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install system dependencies for audio processing + Essentia
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    cmake \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev \
    libfftw3-dev \
    libeigen3-dev \
    libyaml-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libtag1-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip 

# Install build dependencies for madmom first
RUN pip install --no-cache-dir cython>=0.24 numpy>=1.8.1

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install requirements with error handling for Essentia
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Retrying with safer installation..." && \
     pip install --no-cache-dir numpy scipy && \
     pip install --no-cache-dir -r requirements.txt --force-reinstall)

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads temp logs static

# Set environment variables for GPU
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV NVIDIA_VISIBLE_DEVICES=all

# Expose port
EXPOSE 8001

# Run command
CMD ["python", "main.py"]
# Multi-stage Docker build for Face Recognition Service

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as app

# Copy application code from parent directory
COPY ../src /app/src
COPY ../data /app/data
COPY ../models /app/models
COPY ../database /app/database
COPY ../frontend /app/frontend
COPY ../tests /app/tests
COPY ../README.md /app/

# Create necessary directories
RUN mkdir -p data/gallery data/uploads data/raw data/processed data/embeddings \
    models/detector models/embedder models/onnx \
    database logs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application from new location
CMD ["python", "src/api_service.py"]

# Alternative: Use uvicorn directly
# CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000"]

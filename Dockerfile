# Use a lightweight official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and YOLOv8
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install CPU-specific PyTorch to optimize build size for cloud deployment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy model weights and server code
COPY violence_classifier.pt .
COPY yolov8n.pt .
COPY api_server.py .

# Expose port
EXPOSE 8000

# Start FastAPI server using Uvicorn
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

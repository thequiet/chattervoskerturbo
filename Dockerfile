FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    unzip \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git \
    vosk \
    gradio \
    numpy==1.26.4 \
    librosa \
    soundfile \
    git+https://github.com/thequiet/chatterbox.git@faster \
    peft \
    psutil \
    boto3 \
    botocore \
    requests

# Install PyTorch with CUDA support and Triton
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    triton>=2.0.0

# Set working directory
WORKDIR /app

# Copy application code and scripts
COPY app.py /app/app.py
COPY pod_shutdown.py /app/pod_shutdown.py
COPY download_models.sh /app/download_models.sh
COPY setup_network_volume.py /app/setup_network_volume.py
COPY stop_inactive_pod.py /app/stop_inactive_pod.py
COPY start.sh /app/start.sh
RUN chmod +x /app/download_models.sh /app/start.sh

# Create local models directory (for fallback storage)
RUN mkdir -p /app/models

# Expose Gradio port
EXPOSE 7860

# Use the startup script
CMD ["./start.sh"]
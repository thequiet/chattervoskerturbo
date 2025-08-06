# Extend the RunPod PyTorch image
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-test

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Verify and set LD_LIBRARY_PATH for cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git \
    faster-whisper==1.1.0 \
    ctranslate2==4.4.0 \
    vosk \
    gradio \
    numpy==1.26.4 \
    torchaudio==2.8.0+cu128 \
    librosa \
    chatterbox-tts \
    peft \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Set working directory
WORKDIR /app

# Copy application code and scripts
COPY app.py /app/app.py
COPY download_models.sh /app/download_models.sh
RUN chmod +x /app/download_models.sh

# Expose Gradio port
EXPOSE 7860

# Run the model download script and start Gradio
CMD ["/bin/bash", "-c", "./download_models.sh && python3 app.py"]
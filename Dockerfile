# Extend the RunPod PyTorch image
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-test

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    unzip \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN 9.1.0 for CUDA 12.x
RUN curl -O https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz \
    && tar -xf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz \
    && cp cudnn-*/lib/* /usr/local/cuda/lib64/ \
    && cp cudnn-*/include/* /usr/local/cuda/include/ \
    && chmod a+r /usr/local/cuda/lib64/libcudnn*.so* \
    && rm -rf cudnn-linux-x86_64-9.1.0.70_cuda12-archive.tar.xz cudnn-*

# Set LD_LIBRARY_PATH for cuDNN
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python dependencies sequentially
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir ctranslate2==4.4.0
RUN pip install --no-cache-dir faster-whisper==1.1.0
RUN pip install --no-cache-dir \
    git+https://github.com/openai/whisper.git \
    vosk \
    gradio \
    librosa \
    chatterbox-tts \
    peft

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
#!/bin/bash
# Download VOSK model
if [ ! -d "/app/models/vosk-model-en-us-0.22" ]; then
    echo "Downloading VOSK model..."
    curl -L -o vosk-model.tar.gz https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.tar.gz
    mkdir -p /app/models
    tar -xzf vosk-model.tar.gz -C /app/models
    rm vosk-model.tar.gz
fi
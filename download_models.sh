#!/bin/bash
# Download VOSK model
if [ ! -d "/app/models/vosk-model-en-us-0.22" ]; then
    echo "Downloading VOSK model..."
    curl -L -o vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
    if [ $? -ne 0 ] || [ ! -f vosk-model.zip ]; then
        echo "Error: Failed to download VOSK model"
        exit 1
    fi
    # Check file size (should be ~1.4 GB, i.e., > 1GB)
    FILE_SIZE=$(stat -c%s vosk-model.zip 2>/dev/null || stat -f%z vosk-model.zip 2>/dev/null)
    if [ "$FILE_SIZE" -lt 1000000000 ]; then
        echo "Error: Downloaded file is too small ($FILE_SIZE bytes), expected ~1.4 GB"
        rm -f vosk-model.zip
        exit 1
    fi
    # Verify zip integrity
    unzip -t vosk-model.zip > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: Downloaded file is not a valid zip archive"
        rm -f vosk-model.zip
        exit 1
    fi
    mkdir -p /app/models
    unzip vosk-model.zip -d /app/models
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract VOSK model"
        rm -f vosk-model.zip
        exit 1
    fi
    rm -f vosk-model.zip
    echo "VOSK model downloaded and extracted successfully"
else
    echo "VOSK model already exists at /app/models/vosk-model-en-us-0.22"
fi
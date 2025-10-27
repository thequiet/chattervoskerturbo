#!/bin/bash
# Download models with network volume support and local fallback

# Ensure we exit on any error
set -e

# Log all output to a file
exec > >(tee -a /tmp/model_download.log) 2>&1

echo "Starting model download process... (PID: $$ at $(date))"

# Configuration
NETWORK_STORAGE="/network-storage"
LOCAL_MODELS_DIR="/app/models"
NETWORK_MODELS_DIR="${NETWORK_STORAGE}/models"
#VOSK_MODEL_IDENTIFIER="vosk-model-en-us-0.22" # FULL
VOSK_MODEL_IDENTIFIER="vosk-model-en-us-0.22-lgraph"
VOSK_MODEL_NAME="vosk-model-en-us-0.22"

WHISPER_CACHE_DIR="${NETWORK_STORAGE}/whisper-cache"
CHATTERBOX_CACHE_DIR="${NETWORK_STORAGE}/chatterbox-cache"
LOCAL_WHISPER_CACHE="/root/.cache/whisper"
LOCAL_CHATTERBOX_CACHE="/root/.cache/huggingface"

# VOSK model configuration
VOSK_MODEL_URL="https://alphacephei.com/vosk/models/${VOSK_MODEL_IDENTIFIER}.zip"
# MIN_FILE_SIZE=1800000000  # ~1.8GB (LARGE)
MIN_FILE_SIZE=120000000  # 120 MB (MEDIUM)
REQUIRED_SPACE=5000000000  # 5GB
MAX_RETRIES=3
RETRY_DELAY=30
LOCK_FILE="/tmp/model_download.lock"

# Function to check available disk space
check_disk_space() {
    local target_dir=$1
    local required_space=$2
    
    if [ ! -d "$target_dir" ]; then
        mkdir -p "$target_dir" 2>/dev/null || true
    fi
    
    local available_space=$(df -B1 "$target_dir" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    if [ "$available_space" -lt "$required_space" ]; then
        echo "WARNING: Insufficient disk space in $target_dir. Need $required_space bytes, have $available_space bytes."
        return 1
    fi
    return 0
}

# Function to setup network storage
setup_network_storage() {
    echo "Setting up network storage..."
    
    # Try to mount network volume if not already mounted
    if [ ! -d "$NETWORK_STORAGE" ] || [ ! -w "$NETWORK_STORAGE" ]; then
        echo "Network storage not available, attempting to mount..."
        if python /app/setup_network_volume.py mount; then
            echo "✓ Network storage mounted successfully"
        else
            echo "✗ Failed to mount network storage"
            return 1
        fi
    else
        echo "✓ Network storage already available"
    fi
    
    # Create necessary directories on network storage
    mkdir -p "$NETWORK_MODELS_DIR"
    mkdir -p "$WHISPER_CACHE_DIR"
    mkdir -p "$CHATTERBOX_CACHE_DIR"
    
    return 0
}

# Function to setup local fallback
setup_local_fallback() {
    echo "Setting up local storage fallback..."
    mkdir -p "$LOCAL_MODELS_DIR"
    mkdir -p "$(dirname $LOCAL_WHISPER_CACHE)"
    mkdir -p "$(dirname $LOCAL_CHATTERBOX_CACHE)"
}

# Function to determine storage strategy
determine_storage_strategy() {
    if setup_network_storage; then
        echo "Using network storage strategy"
        MODELS_DIR="$NETWORK_MODELS_DIR"
        USE_NETWORK_STORAGE=true
        
        # Set up cache directory symlinks to network storage
        if [ ! -L "$LOCAL_WHISPER_CACHE" ]; then
            rm -rf "$LOCAL_WHISPER_CACHE" 2>/dev/null || true
            ln -sf "$WHISPER_CACHE_DIR" "$LOCAL_WHISPER_CACHE"
        fi
        
        if [ ! -L "$LOCAL_CHATTERBOX_CACHE" ]; then
            rm -rf "$LOCAL_CHATTERBOX_CACHE" 2>/dev/null || true
            ln -sf "$CHATTERBOX_CACHE_DIR" "$LOCAL_CHATTERBOX_CACHE"
        fi
        
    else
        echo "Using local storage strategy (network storage unavailable)"
        setup_local_fallback
        MODELS_DIR="$LOCAL_MODELS_DIR"
        USE_NETWORK_STORAGE=false
    fi
    
    echo "Models will be stored in: $MODELS_DIR"
}

# Function to safely create lock
create_lock() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if (set -C; echo $$ > "${LOCK_FILE}") 2>/dev/null; then
            echo "Successfully created lock file for this process (PID: $$)"
            return 0
        fi
        
        # Check if existing lock is from a dead process
        if [ -f "${LOCK_FILE}" ]; then
            local existing_pid=$(cat "${LOCK_FILE}" 2>/dev/null || echo "")
            if [ -n "$existing_pid" ] && ! kill -0 "$existing_pid" 2>/dev/null; then
                echo "Removing stale lock from dead process (PID: $existing_pid)"
                rm -f "${LOCK_FILE}" 2>/dev/null || true
                continue
            fi
        fi
        
        echo "Lock creation attempt $attempt/$max_attempts failed, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "ERROR: Could not create lock file after $max_attempts attempts"
    return 1
}

# Cleanup function to remove lock file on exit
cleanup() {
    echo "Cleaning up lock file..."
    rm -f "${LOCK_FILE}" 2>/dev/null || true
}
trap cleanup EXIT

# Function to download VOSK model
download_vosk_model() {
    local target_dir=$1
    local model_path="${target_dir}/${VOSK_MODEL_NAME}"
    local zip_file="${target_dir}/vosk-model.zip"
    
    echo "Downloading VOSK model to: $model_path"
    
    # Check if model already exists and is complete
    if [ -d "$model_path" ] && [ -f "$model_path/am/final.mdl" ]; then
        echo "✓ VOSK model already exists and appears complete at $model_path"
        return 0
    fi
    
    # Check disk space
    if ! check_disk_space "$target_dir" "$REQUIRED_SPACE"; then
        echo "ERROR: Insufficient disk space for VOSK model download"
        return 1
    fi
    
    # Change to target directory
    cd "$target_dir"
    
    # Clean up any incomplete downloads
    if [ -f "$zip_file" ]; then
        local file_size=$(stat -c%s "$zip_file" 2>/dev/null || stat -f%z "$zip_file" 2>/dev/null)
        echo "Found existing zip file: $zip_file (${file_size} bytes)"
        if [ ! -f "$model_path/am/final.mdl" ]; then
            echo "Model directory incomplete - removing existing zip to start fresh..."
            rm -f "$zip_file"
            rm -rf "$model_path"
        fi
    fi
    
    # Download with multiple fallback attempts
    local download_success=false
    for attempt in $(seq 1 ${MAX_RETRIES}); do
        echo "Download attempt ${attempt}/${MAX_RETRIES}..."
        
        # Check if zip file already exists and is the right size
        if [ -f "$zip_file" ]; then
            local file_size=$(stat -c%s "$zip_file" 2>/dev/null || stat -f%z "$zip_file" 2>/dev/null)
            if [ "$file_size" -ge "${MIN_FILE_SIZE}" ]; then
                echo "Zip file already exists with correct size: ${file_size} bytes"
                download_success=true
                break
            else
                echo "Existing zip file too small (${file_size} bytes), re-downloading..."
                rm -f "$zip_file"
            fi
        fi
        
        # Use curl with resume capability and better settings
        if curl -L -C - --max-time 3600 --retry 3 --retry-delay 5 \
             --connect-timeout 30 --speed-time 60 --speed-limit 50000 \
             -o "$zip_file" "$VOSK_MODEL_URL" 2>> /tmp/model_download.log; then
            
            # Check if download was successful
            if [ -f "$zip_file" ]; then
                local file_size=$(stat -c%s "$zip_file" 2>/dev/null || stat -f%z "$zip_file" 2>/dev/null)
                if [ "$file_size" -ge "${MIN_FILE_SIZE}" ]; then
                    echo "Download successful! File size: ${file_size} bytes"
                    download_success=true
                    break
                else
                    echo "Downloaded file too small (${file_size} bytes), retrying..."
                    rm -f "$zip_file"
                fi
            fi
        fi
        
        echo "Download failed, retrying in ${RETRY_DELAY} seconds..."
        rm -f "$zip_file"
        sleep ${RETRY_DELAY}
    done
    
    # Check if download ultimately failed
    if [ "$download_success" = false ]; then
        echo "ERROR: Failed to download VOSK model after ${MAX_RETRIES} attempts"
        rm -f "$zip_file"
        return 1
    fi
    
    # Verify zip integrity
    echo "Verifying download integrity..."
    if ! unzip -t "$zip_file" > /dev/null; then
        echo "ERROR: Downloaded file is not a valid zip archive"
        rm -f "$zip_file"
        return 1
    fi
    
    echo "Extracting VOSK model..."
    if ! unzip -q "$zip_file" -d "$target_dir"; then
        echo "ERROR: Failed to extract VOSK model"
        rm -f "$zip_file"
        return 1
    fi
    
    # Verify the extraction was successful
    if [ ! -f "$model_path/am/final.mdl" ]; then
        echo "ERROR: Model extraction appears incomplete - key files not found"
        rm -f "$zip_file"
        rm -rf "$model_path"
        return 1
    fi
    
    # Verify model directory size
    local model_size=$(du -sb "$model_path" | cut -f1)
    if [ "$model_size" -lt 1800000000 ]; then
        echo "ERROR: Extracted model directory too small ($model_size bytes)"
        rm -rf "$model_path"
        return 1
    fi
    
    rm -f "$zip_file"
    echo "✓ VOSK model downloaded and extracted successfully to $model_path"
    return 0
}

# Function to setup symlinks for local access
setup_model_symlinks() {
    if [ "$USE_NETWORK_STORAGE" = true ]; then
        echo "Setting up symlinks for local model access..."
        
        # Create local models directory
        mkdir -p "$LOCAL_MODELS_DIR"
        
        # Create symlink to VOSK model on network storage
        local network_model_path="${NETWORK_MODELS_DIR}/${VOSK_MODEL_NAME}"
        local local_model_link="${LOCAL_MODELS_DIR}/${VOSK_MODEL_NAME}"
        
        if [ -d "$network_model_path" ] && [ ! -L "$local_model_link" ]; then
            rm -rf "$local_model_link" 2>/dev/null || true
            ln -sf "$network_model_path" "$local_model_link"
            echo "✓ Created symlink: $local_model_link -> $network_model_path"
        fi
    fi
}

# Main execution
echo "=================================================="
echo "Model Download Script Starting"
echo "Network Storage: $NETWORK_STORAGE"
echo "Local Models: $LOCAL_MODELS_DIR"
echo "=================================================="

# Check for existing lock file and wait if necessary
if [ -f "${LOCK_FILE}" ]; then
    LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
    if [ -n "${LOCK_PID}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
        echo "Another download process is already running (PID: ${LOCK_PID}). Waiting up to 10 minutes..."
        for i in {1..60}; do
            if [ ! -f "${LOCK_FILE}" ]; then
                echo "Previous download process finished. Continuing..."
                break
            fi
            
            # Check if the process is still running
            CURRENT_LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
            if [ -z "${CURRENT_LOCK_PID}" ] || ! kill -0 "${CURRENT_LOCK_PID}" 2>/dev/null; then
                echo "Previous download process no longer running. Removing stale lock..."
                rm -f "${LOCK_FILE}"
                break
            fi
            
            if [ $i -eq 60 ]; then
                echo "ERROR: Timeout waiting for previous download to complete"
                exit 1
            fi
            
            sleep 10
        done
    else
        echo "Removing stale lock file..."
        rm -f "${LOCK_FILE}"
    fi
fi

# Determine storage strategy
determine_storage_strategy

# Check if VOSK model already exists (before acquiring lock)
VOSK_MODEL_PATH="${MODELS_DIR}/${VOSK_MODEL_NAME}"
if [ -d "$VOSK_MODEL_PATH" ] && [ -f "$VOSK_MODEL_PATH/am/final.mdl" ]; then
    echo "✓ VOSK model already exists and appears complete at $VOSK_MODEL_PATH"
    setup_model_symlinks
    echo "✓ Model setup completed successfully"
    exit 0
fi

# Acquire lock for download
echo "Model not found or incomplete - attempting to acquire lock for download..."
if ! create_lock; then
    echo "ERROR: Could not acquire download lock"
    exit 1
fi

# Download VOSK model
echo "Downloading VOSK model..."
if download_vosk_model "$MODELS_DIR"; then
    echo "✓ VOSK model download completed successfully"
else
    echo "✗ VOSK model download failed"
    
    # If using network storage and it failed, try local fallback
    if [ "$USE_NETWORK_STORAGE" = true ]; then
        echo "Attempting local storage fallback..."
        setup_local_fallback
        MODELS_DIR="$LOCAL_MODELS_DIR"
        USE_NETWORK_STORAGE=false
        
        if download_vosk_model "$MODELS_DIR"; then
            echo "✓ VOSK model downloaded to local storage as fallback"
        else
            echo "✗ VOSK model download failed even with local fallback"
            exit 1
        fi
    else
        exit 1
    fi
fi

# Setup symlinks if using network storage
setup_model_symlinks

echo "✓ Model download and setup completed successfully"
echo "Models directory: $MODELS_DIR"
echo "Storage strategy: $([ "$USE_NETWORK_STORAGE" = true ] && echo "Network Storage" || echo "Local Storage")"

#!/bin/bash
# Download VOSK model with improved error handling and resume capability

# Ensure we exit on any error
set -e

# Log all output to a file
exec > >(tee -a /tmp/vosk_download.log) 2>&1

# Configuration
MODEL_NAME="vosk-model-en-us-0.22"
MODEL_IDENTIFIER="vosk-model-en-us-0.22-lgraph"
MODEL_DIR="/app/models"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
ZIP_FILE="vosk-model.zip"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_IDENTIFIER}.zip"
# MIN_FILE_SIZE=1800000000  # ~1.8GB (LARGE)
MIN_FILE_SIZE=120000000  # 120 MB (MEDIUM)
REQUIRED_SPACE=5000000000  # 5GB
MAX_RETRIES=10
RETRY_DELAY=30
LOCK_FILE="/tmp/vosk_download.lock"

echo "Starting VOSK model download process... (PID: $$ at $(date))"
echo "Model: ${MODEL_NAME}"
echo "Target directory: ${MODEL_PATH}"

# Check disk space
AVAILABLE_SPACE=$(df -B1 /app/models | tail -1 | awk '{print $4}' || echo 0)
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "ERROR: Insufficient disk space. Need $REQUIRED_SPACE bytes, have $AVAILABLE_SPACE bytes."
    exit 1
fi

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

# Check if model is already downloaded and extracted (do this first, before locking)
echo "Checking if model already exists..."
if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/am/final.mdl" ]; then
    echo "VOSK model already exists and appears complete at ${MODEL_PATH}"
    echo "Model validation successful - exiting without lock"
    exit 0
fi

# Only proceed with locking if we actually need to download
echo "Model not found or incomplete - attempting to acquire lock for download..."

# Check for existing lock file and wait if necessary
if [ -f "${LOCK_FILE}" ]; then
    LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
    if [ -n "${LOCK_PID}" ] && kill -0 "${LOCK_PID}" 2>/dev/null; then
        echo "Another download process is already running (PID: ${LOCK_PID}). Waiting up to 20 minutes..."
        # Wait for the other process to finish (max 20 minutes)
        for i in {1..120}; do
            if [ ! -f "${LOCK_FILE}" ]; then
                echo "Previous download process finished. Continuing..."
                break
            fi
            
            # Check if the process is still running
            CURRENT_LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
            if [ -z "${CURRENT_LOCK_PID}" ] || ! kill -0 "${CURRENT_LOCK_PID}" 2>/dev/null; then
                echo "Previous download process (PID: ${CURRENT_LOCK_PID}) no longer running. Removing stale lock..."
                rm -f "${LOCK_FILE}"
                break
            fi
            
            sleep 10
            if [ $((i % 6)) -eq 0 ]; then  # Log every minute
                echo "Still waiting for previous download... (${i}/120) - process ${CURRENT_LOCK_PID} is still running"
            fi
        done
        
        # If still locked after 20 minutes, force remove (assume stuck)
        if [ -f "${LOCK_FILE}" ]; then
            FINAL_LOCK_PID=$(cat "${LOCK_FILE}" 2>/dev/null)
            echo "Download process appears stuck after 20 minutes. Force removing lock (PID: ${FINAL_LOCK_PID})"
            rm -f "${LOCK_FILE}"
        fi
        
        # Check again if model was downloaded by the other process
        if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/am/final.mdl" ]; then
            echo "Model was successfully downloaded by another process - exiting"
            exit 0
        fi
    else
        echo "Removing stale lock file (process ${LOCK_PID} no longer running)"
        rm -f "${LOCK_FILE}"
    fi
fi

# Create lock file using atomic operation
if ! create_lock; then
    echo "ERROR: Failed to acquire download lock"
    exit 1
fi

# Check model directory state now that we have the lock
echo "Checking model directory state (with lock acquired)..."
if [ -d "${MODEL_PATH}" ]; then
    if [ -f "${MODEL_PATH}/am/final.mdl" ]; then
        echo "Model was completed by another process while we were waiting - exiting"
        exit 0
    else
        echo "Model directory exists but appears incomplete. Contents:"
        ls -la "${MODEL_PATH}/" 2>/dev/null || echo "Could not list contents"
        echo "Removing incomplete model directory..."
        rm -rf "${MODEL_PATH}"
    fi
fi

echo "Proceeding with download..."
echo "Downloading VOSK model..."

# Create models directory
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"
echo "Working directory: $(pwd)"
echo "Contents of models directory:"
ls -la "${MODEL_DIR}/" 2>/dev/null || echo "Models directory is empty"

# If partial download exists but model directory is incomplete, clean up
if [ -f "${ZIP_FILE}" ]; then
    FILE_SIZE=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null)
    echo "Found existing zip file: ${ZIP_FILE} (${FILE_SIZE} bytes)"
    if [ ! -f "${MODEL_PATH}/am/final.mdl" ]; then
        echo "Model directory incomplete - removing existing zip to start fresh..."
        rm -f "${ZIP_FILE}"
        rm -rf "${MODEL_PATH}"
    fi
fi

# Download with multiple fallback attempts
DOWNLOAD_SUCCESS=false
for attempt in $(seq 1 ${MAX_RETRIES}); do
    echo "Download attempt ${attempt}/${MAX_RETRIES}..."
    
    # Check if zip file already exists and is the right size
    if [ -f "${ZIP_FILE}" ]; then
        FILE_SIZE=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null)
        if [ "$FILE_SIZE" -ge "${MIN_FILE_SIZE}" ]; then
            echo "Zip file already exists with correct size: ${FILE_SIZE} bytes"
            DOWNLOAD_SUCCESS=true
            break
        else
            echo "Existing zip file too small (${FILE_SIZE} bytes), re-downloading..."
            rm -f "${ZIP_FILE}"
        fi
    fi
    
    # Use curl with resume capability and better settings
    curl -L -C - --max-time 3600 --retry 3 --retry-delay 5 \
         --connect-timeout 30 --speed-time 60 --speed-limit 50000 \
         -o "${ZIP_FILE}" "${MODEL_URL}" 2>> /tmp/vosk_download.log
    
    # Check if download was successful
    if [ $? -eq 0 ] && [ -f "${ZIP_FILE}" ]; then
        # Verify file size
        FILE_SIZE=$(stat -c%s "${ZIP_FILE}" 2>/dev/null || stat -f%z "${ZIP_FILE}" 2>/dev/null)
        if [ "$FILE_SIZE" -ge "${MIN_FILE_SIZE}" ]; then
            echo "Download successful! File size: ${FILE_SIZE} bytes"
            DOWNLOAD_SUCCESS=true
            break
        else
            echo "Downloaded file too small (${FILE_SIZE} bytes), retrying..."
            rm -f "${ZIP_FILE}"
        fi
    else
        echo "Download failed, retrying in ${RETRY_DELAY} seconds..."
        rm -f "${ZIP_FILE}"
        sleep ${RETRY_DELAY}
    fi
done

# Check if download ultimately failed
if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo "Error: Failed to download VOSK model after ${MAX_RETRIES} attempts"
    rm -f "${ZIP_FILE}"
    exit 1
fi

# Verify zip integrity
echo "Verifying download integrity..."
unzip -t "${ZIP_FILE}" > /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Downloaded file is not a valid zip archive"
    rm -f "${ZIP_FILE}"
    exit 1
fi

echo "Extracting VOSK model..."
unzip -q "${ZIP_FILE}" -d "${MODEL_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract VOSK model"
    rm -f "${ZIP_FILE}"
    exit 1
fi

# Verify the extraction was successful by checking for a key model file
if [ ! -f "${MODEL_PATH}/am/final.mdl" ] && [ ! -f "${MODEL_PATH}/conf/model.conf" ]; then
    echo "Error: Model extraction appears incomplete - key files not found"
    rm -f "${ZIP_FILE}"
    rm -rf "${MODEL_PATH}"
    exit 1
fi

# Verify model directory size
MODEL_SIZE=$(du -sb "${MODEL_PATH}" | cut -f1)
if [ "$MODEL_SIZE" -lt 1800000000 ]; then
    echo "ERROR: Extracted model directory too small ($MODEL_SIZE bytes)"
    rm -rf "${MODEL_PATH}"
    exit 1
fi

rm -f "${ZIP_FILE}"
echo "VOSK model downloaded and extracted successfully"

# Final validation
echo "Final validation - checking model files..."
if [ -f "${MODEL_PATH}/am/final.mdl" ]; then
    echo "✓ Model validation successful: ${MODEL_PATH}/am/final.mdl found"
else
    echo "✗ Model validation failed: key files not found"
    exit 1
fi
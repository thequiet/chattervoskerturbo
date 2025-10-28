#!/bin/bash
# Startup script for ChatterVosker application with optimized package management

# Safer bash options
set -euo pipefail
set -x  # Print commands as they are executed

# Enable synchronous CUDA error reporting (can override by pre-setting variable)
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
echo "CUDA_LAUNCH_BLOCKING is set to $CUDA_LAUNCH_BLOCKING"

echo "=================================================="
echo "ChatterVosker Container Starting (Optimized)"
echo "Timestamp: $(date)"
echo "Container PID: $$"
echo "Working Directory: $(pwd)"
echo "User: $(whoami)"
echo "Python version: $(python --version)"
echo "=================================================="

# Function to handle graceful shutdown
cleanup() {
    echo "Received shutdown signal, cleaning up..."
    # Try to upload logs before killing processes (best-effort)
    upload_logs_on_shutdown "trap_cleanup"
    # Kill background processes
    if [ ! -z "${MONITOR_PID:-}" ]; then
        echo "Stopping inactivity monitor (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null || true
    fi
    if [ ! -z "${APP_PID:-}" ]; then
        echo "Stopping application (PID: $APP_PID)..."
        kill $APP_PID 2>/dev/null || true
    fi
    echo "Cleanup complete."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Helper: upload logs to S3 via centralized Python routine (best-effort)
upload_logs_on_shutdown() {
    local reason=${1:-"shutdown"}
    echo "Uploading logs to S3 (reason: $reason)..."
    # Ensure logs are flushed
    sync || true
    sleep 1 || true
    # Provide a list of likely logs
    SHUTDOWN_REASON="$reason" \
    EXTRA_LOGS="/app/app.log,/app/inactivity_monitor.log,/tmp/model_download.log,/tmp/package_install.log" \
    python pod_shutdown.py || true
}

# Check resources
echo "Checking system resources..."
# Note: We'll check space after determining storage strategy
# Check memory using /proc/meminfo (free command not available in container)
AVAILABLE_MEMORY_KB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 4000001)
AVAILABLE_MEMORY=$((AVAILABLE_MEMORY_KB * 1024))
if [ "$AVAILABLE_MEMORY" -lt 4000000000 ]; then
    echo "ERROR: Insufficient memory. Need 4GB, have $AVAILABLE_MEMORY bytes."
    exit 1
fi

# Step 0: Setup network storage (if possible)
echo "Step 0: Setting up storage strategy..."
echo "Attempting to setup network storage..."
NETWORK_STORAGE_PATH="${RUNPOD_MOUNT_PATH:-/network-storage}"
echo "Network storage path: $NETWORK_STORAGE_PATH"
if python setup_network_volume.py mount; then
    echo "✓ Network storage available"
    # Check space on network storage
    AVAILABLE_SPACE=$(df -B1 "$NETWORK_STORAGE_PATH" 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    if [ "$AVAILABLE_SPACE" -lt 5000000000 ]; then
        echo "WARNING: Insufficient space on network storage. Need 5GB, have $AVAILABLE_SPACE bytes."
        echo "Will fall back to local storage if needed."
    fi
else
    echo "Network storage not available, will use local storage"
    # Check space on local storage
    AVAILABLE_SPACE=$(df -B1 /app 2>/dev/null | tail -1 | awk '{print $4}' || echo 0)
    if [ "$AVAILABLE_SPACE" -lt 8000000000 ]; then
        echo "ERROR: Insufficient disk space. Need 8GB, have $AVAILABLE_SPACE bytes."
        exit 1
    fi
fi

# Step 1: Install heavy packages (if not in image)
echo "Step 1: Installing heavy Python packages..."
if [ -f "/app/install_heavy_packages.sh" ]; then
    echo "Running heavy package installation..."
    if ! ./install_heavy_packages.sh 2>&1 | tee /tmp/package_install.log; then
        PACKAGE_EXIT_CODE=$?
        echo "ERROR: Heavy package installation failed with exit code: $PACKAGE_EXIT_CODE"
        echo "Checking package installation logs..."
        tail -20 /tmp/package_install.log 2>/dev/null || echo "No package log found"
        echo "Exiting due to package installation failure."
        exit 1
    fi
    echo "✓ Heavy packages installed successfully"
else
    echo "No heavy package installer found, assuming packages are in image"
fi

# Step 2: Download/setup models (with network storage support)
echo "Step 2: Setting up models with network storage support..."
echo "Current directory contents:"
ls -la
echo "Checking if download script exists and is executable:"
ls -la download_models.sh

# Run the download script and exit on failure
if ! ./download_models.sh; then
    DOWNLOAD_EXIT_CODE=$?
    echo "ERROR: Model setup failed with exit code: $DOWNLOAD_EXIT_CODE"
    echo "Checking logs and directory state..."
    ls -la /app/models/ 2>/dev/null || echo "Local models directory does not exist"
    ls -la /workspace/models/ 2>/dev/null || echo "Workspace models directory does not exist"
    ls -la "$NETWORK_STORAGE_PATH/models/" 2>/dev/null || echo "Network models directory does not exist"
    tail -20 /tmp/model_download.log 2>/dev/null || echo "No download log found"
    echo "Exiting due to model setup failure."
    exit 1
fi
echo "✓ Model setup completed successfully"

# Step 3: Start the inactivity monitor in the background
echo "Step 3: Starting inactivity monitor..."
echo "Checking if stop_inactive_pod.py exists:"
ls -la stop_inactive_pod.py

echo "Testing Python import capabilities..."
if python -c "import sys; print('Python executable:', sys.executable)"; then
    echo "✓ Python basic test passed"
else
    echo "✗ Python basic test failed"
    exit 1
fi

if python -c "import logging; print('✓ logging module works')"; then
    echo "✓ Python logging test passed"
else
    echo "✗ Python logging test failed"
    exit 1
fi

# Try to start monitor but don't fail if it doesn't work
if python stop_inactive_pod.py & then
    MONITOR_PID=$!
    echo "✓ Inactivity monitor started (PID: $MONITOR_PID)"
    sleep 2  # Give it a moment to start
    if ! kill -0 $MONITOR_PID 2>/dev/null; then
        echo "WARNING: Inactivity monitor process died immediately"
        MONITOR_PID=""
    fi
else
    echo "WARNING: Failed to start inactivity monitor"
    MONITOR_PID=""
fi

# Step 4: Start the main application
echo "Step 4: Starting main application..."
echo "Checking if app.py exists:"
ls -la app.py

echo "Testing critical Python imports..."
IMPORT_TESTS_PASSED=true

if python -c "import torch; print('✓ torch version:', torch.__version__)"; then
    echo "✓ PyTorch import test passed"
else
    echo "✗ PyTorch import test failed"
    IMPORT_TESTS_PASSED=false
fi

if python -c "import gradio; print('✓ gradio version:', gradio.__version__)"; then
    echo "✓ Gradio import test passed"
else
    echo "✗ Gradio import test failed"
    IMPORT_TESTS_PASSED=false
fi

if python -c "import whisper; print('✓ whisper imported successfully')"; then
    echo "✓ Whisper import test passed"
else
    echo "✗ Whisper import test failed"
    IMPORT_TESTS_PASSED=false
fi

if python -c "import vosk; print('✓ vosk imported successfully')"; then
    echo "✓ VOSK import test passed"
else
    echo "✗ VOSK import test failed"
    IMPORT_TESTS_PASSED=false
fi

if python -c "from chatterbox.tts import ChatterboxTTS; print('✓ chatterbox imported successfully')"; then
    echo "✓ Chatterbox import test passed"
else
    echo "✗ Chatterbox import test failed"
    IMPORT_TESTS_PASSED=false
fi

if [ "$IMPORT_TESTS_PASSED" = false ]; then
    echo "ERROR: Critical imports failed. Exiting."
    upload_logs_on_shutdown "import_failure"
    exit 1
fi

echo "Application will be available at http://0.0.0.0:7860"
echo "Logs will be written to /app/app.log"
echo "=================================================="

# Start the application and capture its PID
echo "Starting application..."
# Start python directly so APP_PID reflects python, not tee.
python app.py >> app.log 2>&1 &
APP_PID=$!
echo "✓ Main application started (PID: $APP_PID)"
# Stream logs to console in background for visibility
tail -n +1 -F app.log &
TAIL_PID=$!

# Give the app a moment to start and check if it's still running
sleep 5
if ! kill -0 $APP_PID 2>/dev/null; then
    echo "ERROR: Application process died immediately!"
    echo "Application log contents:"
    cat app.log 2>/dev/null || echo "No app.log found"
    echo "Container will keep running for debugging purposes..."
    upload_logs_on_shutdown "app_died_early"
    while true; do
        echo "Container is alive for debugging. Sleeping..."
        sleep 30
    done
fi
echo "✓ Application appears to be running successfully"

# Wait for the application to finish (this keeps the container running)
if [ ! -z "$APP_PID" ]; then
    echo "Waiting for application to finish..."
    wait $APP_PID
    APP_EXIT_CODE=$?
    echo "Application exited with code: $APP_EXIT_CODE"
    # Upload logs after normal exit
    upload_logs_on_shutdown "app_exit_${APP_EXIT_CODE}"
else
    echo "No application PID to wait for"
    APP_EXIT_CODE=1
fi

# Clean up the monitor process
if [ ! -z "$MONITOR_PID" ]; then
    echo "Cleaning up inactivity monitor..."
    kill $MONITOR_PID 2>/dev/null || true
else
    echo "No monitor PID to clean up"
fi

# Clean up tailer
if [ ! -z "${TAIL_PID:-}" ]; then
    echo "Stopping log tailer..."
    kill $TAIL_PID 2>/dev/null || true
fi

exit $APP_EXIT_CODE

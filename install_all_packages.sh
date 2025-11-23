#!/bin/bash
# Install ALL packages to network storage with comprehensive fallback

set -e

echo "Installing ALL packages to network storage..."

# Configuration
NETWORK_STORAGE="${RUNPOD_MOUNT_PATH:-/network-storage}"
NETWORK_PACKAGES_DIR="${NETWORK_STORAGE}/python-packages"
LOCAL_PACKAGES_DIR="/app/python-packages"

# All packages (both heavy and light)
ALL_PACKAGES=(
    "gradio"
    "fastapi"
    "uvicorn[standard]"
    "numpy==1.26.4"
    "librosa" 
    "psutil"
    "boto3"
    "botocore"
    "requests"
    "git+https://github.com/openai/whisper.git"
    "vosk"
    "git+https://github.com/thequiet/chatterbox.git@faster"
    "peft"
    "torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    "triton>=2.0.0"
)

# Function to setup Python path for network storage
setup_python_path() {
    local packages_dir=$1
    local site_packages_dir="${packages_dir}/site-packages"
    
    mkdir -p "$site_packages_dir"
    
    # Add to Python path
    export PYTHONPATH="${site_packages_dir}:${PYTHONPATH:-}"
    echo "PYTHONPATH=${PYTHONPATH}" >> /etc/environment
    
    # Create pip install target
    export PIP_TARGET="$site_packages_dir"
    
    echo "✓ Python path configured for: $site_packages_dir"
    echo "PYTHONPATH: $PYTHONPATH"
}

# Function to test package installation location
test_package_installation() {
    echo "Testing package installation..."
    
    # Install a test package
    if pip install --target "$PIP_TARGET" --no-deps setuptools 2>/dev/null; then
        echo "✓ Package installation test successful"
        return 0
    else
        echo "✗ Package installation test failed"
        return 1
    fi
}

# Main execution
echo "=================================================="
echo "Complete Package Installation to Network Storage"
echo "Network Storage: $NETWORK_STORAGE"
echo "=================================================="

# Determine storage strategy
if [ -d "$NETWORK_STORAGE" ] && [ -w "$NETWORK_STORAGE" ]; then
    echo "✓ Network storage available"
    PACKAGES_DIR="$NETWORK_PACKAGES_DIR" 
    USE_NETWORK=true
else
    echo "Network storage not available, using local storage"
    PACKAGES_DIR="$LOCAL_PACKAGES_DIR"
    USE_NETWORK=false
fi

# Setup Python path
setup_python_path "$PACKAGES_DIR"

# Test installation capability
if ! test_package_installation; then
    echo "Installation test failed, trying alternative method..."
    
    if [ "$USE_NETWORK" = true ]; then
        echo "Falling back to local storage..."
        PACKAGES_DIR="$LOCAL_PACKAGES_DIR"
        setup_python_path "$PACKAGES_DIR"
        USE_NETWORK=false
        
        if ! test_package_installation; then
            echo "ERROR: Cannot install packages to any location"
            exit 1
        fi
    else
        echo "ERROR: Package installation not working"
        exit 1
    fi
fi

# Install all packages
echo "Installing packages to: $PACKAGES_DIR"
FAILED_PACKAGES=()
INSTALLED_PACKAGES=()

for package in "${ALL_PACKAGES[@]}"; do
    echo "Installing: $package"
    
    if pip install --target "$PIP_TARGET" $package; then
        INSTALLED_PACKAGES+=("$package")
        echo "✓ Installed: $package"
    else
        FAILED_PACKAGES+=("$package")
        echo "✗ Failed: $package"
    fi
done

# Create startup script for Python path
cat > /app/set_python_path.sh << EOF
#!/bin/bash
# Set Python path for installed packages
export PYTHONPATH="${PACKAGES_DIR}/site-packages:\${PYTHONPATH:-}"
export PIP_TARGET="${PACKAGES_DIR}/site-packages"
EOF
chmod +x /app/set_python_path.sh

# Summary
echo "=================================================="
echo "Installation Summary"
echo "=================================================="
echo "Storage: $([ "$USE_NETWORK" = true ] && echo "Network" || echo "Local")"
echo "Location: $PACKAGES_DIR"
echo "Successfully Installed (${#INSTALLED_PACKAGES[@]}):"
for package in "${INSTALLED_PACKAGES[@]}"; do
    echo "  ✓ $package"
done

if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    echo "Failed Installations (${#FAILED_PACKAGES[@]}):"
    for package in "${FAILED_PACKAGES[@]}"; do
        echo "  ✗ $package"
    done
fi

# Test imports
echo "Testing critical imports..."
source /app/set_python_path.sh

CRITICAL_IMPORTS=("torch" "whisper" "vosk" "gradio")
IMPORT_FAILURES=()

for import_name in "${CRITICAL_IMPORTS[@]}"; do
    if python -c "import $import_name" 2>/dev/null; then
        echo "  ✓ $import_name"
    else
        echo "  ✗ $import_name"
        IMPORT_FAILURES+=("$import_name")
    fi
done

# Test chatterbox specifically with its full import path
echo "Testing chatterbox.tts import..."
if python -c "from chatterbox.tts import ChatterboxTTS; print('✓ chatterbox.tts imported successfully')" 2>/dev/null; then
    echo "  ✓ chatterbox.tts"
else
    echo "  ✗ chatterbox.tts"
    IMPORT_FAILURES+=("chatterbox.tts")
fi

if [ ${#IMPORT_FAILURES[@]} -gt 0 ]; then
    echo "ERROR: Critical imports failed: ${IMPORT_FAILURES[*]}"
    exit 1
fi

echo "✓ All packages installed and verified successfully!"
echo "Use 'source /app/set_python_path.sh' to set Python path in other scripts"

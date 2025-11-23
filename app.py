import os

# Optional CUDA debug controls (set via *_OVERRIDE env vars to avoid clobbering container defaults)
cuda_launch_blocking_override = os.environ.get("CUDA_LAUNCH_BLOCKING_OVERRIDE")
if cuda_launch_blocking_override is not None:
    os.environ["CUDA_LAUNCH_BLOCKING"] = cuda_launch_blocking_override

torch_use_cuda_dsa_override = os.environ.get("TORCH_USE_CUDA_DSA_OVERRIDE")
if torch_use_cuda_dsa_override is not None:
    os.environ["TORCH_USE_CUDA_DSA"] = torch_use_cuda_dsa_override

torch_disable_addr2line_override = os.environ.get("TORCH_DISABLE_ADDR2LINE_OVERRIDE")
if torch_disable_addr2line_override is not None:
    os.environ["TORCH_DISABLE_ADDR2LINE"] = torch_disable_addr2line_override

import gradio as gr
import whisper
from vosk import Model, KaldiRecognizer
import json
import wave
import numpy as np
import random
import torch
import torchaudio
import soundfile as sf
import logging
import traceback
import sys
import subprocess
import tempfile
import signal
import uuid
import asyncio
import zipfile
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from chatterbox.tts import ChatterboxTTS
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from pod_shutdown import perform_pod_shutdown
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)
from pydantic import BaseModel
import uvicorn

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if cuda_launch_blocking_override is not None:
    logger.info(f"CUDA_LAUNCH_BLOCKING override applied: {cuda_launch_blocking_override}")

if torch_use_cuda_dsa_override is not None:
    logger.info(f"TORCH_USE_CUDA_DSA override applied: {torch_use_cuda_dsa_override}")

if torch_disable_addr2line_override is not None:
    logger.info(f"TORCH_DISABLE_ADDR2LINE override applied: {torch_disable_addr2line_override}")

# Set up specific loggers for different components
gradio_logger = logging.getLogger('gradio')
gradio_logger.setLevel(logging.DEBUG)

# Custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

logger.info("="*50)
logger.info("Starting ChatterVosker Application")
logger.info(f"Python version: {sys.version}")
logger.info(f"Start time: {datetime.now()}")
logger.info("="*50)

# Configuration settings
SKIP_AUDIO_CONVERSION = os.environ.get("SKIP_AUDIO_CONVERSION", "true").lower() in ("true", "1", "yes")
AUDIO_NORMALIZATION_ENABLED = os.environ.get("AUDIO_NORMALIZATION_ENABLED", "true").lower() in ("true", "1", "yes")
AUDIO_TARGET_LUFS = float(os.environ.get("AUDIO_TARGET_LUFS", "-23.0"))
AUDIO_MAX_PEAK_DB = float(os.environ.get("AUDIO_MAX_PEAK_DB", "-1.0"))
AUDIO_LIMITER_THRESHOLD_DB = float(os.environ.get("AUDIO_LIMITER_THRESHOLD_DB", "-3.0"))
AUDIO_LIMITER_RATIO = float(os.environ.get("AUDIO_LIMITER_RATIO", "8.0"))
PROMPT_TARGET_SAMPLE_RATE = 24000
PROMPT_EXPECTED_CHANNELS = 1
PROMPT_SANITIZED_DIR = "/tmp/chatterbox_prompts"
AUDIO_OUTPUT_DIR = "/app/audio/output"

HEARSAID_API_HOST = os.environ.get("HEARSAID_API_HOST", "0.0.0.0")
HEARSAID_API_PORT = int(os.environ.get("HEARSAID_API_PORT", "7861"))
HEARSAID_API_BASE_PATH = os.environ.get("HEARSAID_API_BASE_PATH", "/hearsaid-api") or "/hearsaid-api"
HEARSAID_API_BASE_PATH = "/" + HEARSAID_API_BASE_PATH.strip("/")
HEARSAID_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
HEARSAID_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".zip"}

logger.info(f"Skip audio conversion: {SKIP_AUDIO_CONVERSION}")
logger.info(f"Audio normalization enabled: {AUDIO_NORMALIZATION_ENABLED}")
logger.info(f"Audio target LUFS: {AUDIO_TARGET_LUFS}")
logger.info(f"Audio max peak: {AUDIO_MAX_PEAK_DB} dB")
logger.info(f"Audio limiter threshold: {AUDIO_LIMITER_THRESHOLD_DB} dB")
logger.info(f"Audio limiter ratio: {AUDIO_LIMITER_RATIO}:1")

# Initialize S3 upload thread pool
S3_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="s3-upload")
S3_UPLOAD_FUTURES = queue.Queue()
logger.info("Initialized S3 upload thread pool with 5 workers")

# Load models
logger.info("Initializing models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Optimize Triton for your GPU
if torch.cuda.is_available():
    logger.info("Setting up CUDA optimizations...")
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
    logger.info(f"GPU Memory before loading: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")

# Check Triton availability
try:
    import triton
    logger.info(f"✓ Triton is available: {triton.__version__}")
    triton_available = True
except ImportError as e:
    logger.warning(f"✗ Triton is not installed: {e}")
    triton_available = False

logger.info(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"✓ CUDA version: {torch.version.cuda}")

# Load Whisper model with error handling and network storage cache
try:
    logger.info("Loading Whisper model...")
    # Whisper will use cache directory from environment or default ~/.cache/whisper
    # The download_models.sh script sets up symlinks for cache redirection
    
    # Set download root to prevent network access if model is already cached
    whisper_cache_dir = os.environ.get("WHISPER_CACHE_DIR", os.path.expanduser("~/.cache/whisper"))
    logger.info(f"Whisper cache directory: {whisper_cache_dir}")
    
    # Try to load with cache, fallback to download if needed
    try:
        whisper_model = whisper.load_model("turbo", device=device, download_root=whisper_cache_dir)
        logger.info("✓ Whisper model loaded successfully")
    except Exception as download_error:
        logger.warning(f"Failed to download Whisper model (network issue): {download_error}")
        logger.info("Attempting to load from existing cache...")
        
        # Try to find cached model
        model_name = "turbo"
        expected_cache_file = os.path.join(whisper_cache_dir, f"{model_name}.pt")
        
        if os.path.exists(expected_cache_file):
            logger.info(f"Found cached model at: {expected_cache_file}")
            whisper_model = whisper.load_model("turbo", device=device, download_root=whisper_cache_dir)
            logger.info("✓ Whisper model loaded from cache")
        else:
            logger.error(f"No cached Whisper model found at: {expected_cache_file}")
            logger.error("Whisper functionality will be unavailable")
            whisper_model = None
            
except Exception as e:
    logger.error(f"✗ Failed to load Whisper model: {e}")
    logger.error(traceback.format_exc())
    logger.error("Whisper functionality will be unavailable")
    whisper_model = None

# Load VOSK model with error handling and network storage support
vosk_model = None
network_storage_path = os.environ.get("RUNPOD_MOUNT_PATH", "/network-storage")
VOSK_MODEL_DIR_CANDIDATES = [
    "vosk-model-en-us-0.22-lgraph",
    "vosk-model-en-us-0.22",
]
VOSK_SEARCH_LOCATIONS = [
    ("workspace", "/workspace/models"),
    ("network storage", f"{network_storage_path}/models"),
    ("local storage", "/app/models"),
]


def resolve_vosk_model_path():
    for location_name, root_dir in VOSK_SEARCH_LOCATIONS:
        for model_dir in VOSK_MODEL_DIR_CANDIDATES:
            candidate_path = os.path.join(root_dir, model_dir)
            if os.path.exists(os.path.join(candidate_path, "am", "final.mdl")):
                logger.info(f"Using VOSK model from {location_name}: {candidate_path}")
                return candidate_path
    return None


vosk_model_path = resolve_vosk_model_path()

if vosk_model_path:
    logger.info(f"Loading VOSK model from: {vosk_model_path}")
    try:
        vosk_model = Model(vosk_model_path)
        logger.info("✓ VOSK model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load VOSK model: {e}")
        logger.error(traceback.format_exc())
        logger.warning("VOSK transcription will be disabled.")
        vosk_model = None
else:
    searched_paths = [
        os.path.join(root_dir, model_dir)
        for _, root_dir in VOSK_SEARCH_LOCATIONS
        for model_dir in VOSK_MODEL_DIR_CANDIDATES
    ]
    logger.warning(
        "VOSK model not found. Searched paths: %s. Attempting to download...",
        ", ".join(searched_paths)
    )
    try:
        result = subprocess.run(["/app/download_models.sh"], capture_output=True, text=True)
        if result.returncode == 0:
            # Check again for model after download (check in priority order)
            vosk_model_path = resolve_vosk_model_path()

            if vosk_model_path:
                logger.info(f"VOSK model downloaded successfully. Loading from {vosk_model_path}...")
                vosk_model = Model(vosk_model_path)
                logger.info("✓ VOSK model loaded successfully")
            else:
                logger.error("VOSK model download completed but model files not found")
                logger.warning("VOSK transcription will be disabled.")
        else:
            logger.error(f"Failed to download VOSK model: {result.stderr}")
            logger.warning("VOSK transcription will be disabled.")
    except Exception as e:
        logger.error(f"✗ Failed to download/load VOSK model: {e}")
        logger.error(traceback.format_exc())
        logger.warning("VOSK transcription will be disabled.")

# Load Chatterbox model with error handling and network storage cache
try:
    logger.info("Loading Chatterbox TTS model...")
    # Chatterbox will use cache directory from environment or default ~/.cache/huggingface
    # The download_models.sh script sets up symlinks for cache redirection
    chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("✓ Chatterbox model loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load Chatterbox model: {e}")
    logger.error(traceback.format_exc())
    raise

if torch.cuda.is_available():
    logger.info(f"GPU Memory after loading models: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")

logger.info("All available models loaded successfully!")

# Guard Chatterbox generations to prevent concurrent CUDA access, which can trigger device asserts
CHATTERBOX_GENERATION_LOCK = threading.Lock()

def save_waveform_with_soundfile(path, waveform, sample_rate, subtype=None):
    """Save a waveform tensor to disk using soundfile.

    Args:
        path (str): Destination file path.
        waveform (torch.Tensor): Audio tensor shaped (channels, samples) or (samples,).
        sample_rate (int): Sample rate for the audio file.
        subtype (str | None): Optional soundfile subtype (e.g., 'PCM_16', 'FLOAT').
    """
    # Ensure tensor is detached on CPU
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Clamp to valid audio range just in case processing introduced overs
    waveform = torch.clamp(waveform, -1.0, 1.0)

    # Convert to (samples, channels) layout for soundfile
    data = waveform.transpose(0, 1).numpy()

    write_kwargs = {"subtype": subtype} if subtype else {}
    sf.write(path, data, sample_rate, **write_kwargs)


def ensure_prompt_audio_format(audio_path, target_sr=PROMPT_TARGET_SAMPLE_RATE):
    """Ensure audio prompt matches expected mono 24kHz PCM16 format.

    Returns:
        tuple[str, str | None]: (path to use, temp path to clean up if created)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio prompt not found: {audio_path}")

    info = None
    bits_per_sample = None

    # Gather metadata using torchaudio (if available) or soundfile as fallback
    try:
        if hasattr(torchaudio, "info"):
            info = torchaudio.info(audio_path)
        elif hasattr(torchaudio.backend, "sox_io_backend") and hasattr(torchaudio.backend.sox_io_backend, "info"):
            info = torchaudio.backend.sox_io_backend.info(audio_path)
        if info is not None:
            bits_per_sample = getattr(info, "bits_per_sample", None)
            logger.info(
                "Audio prompt metadata - path: %s, sample_rate: %s, channels: %s, bits_per_sample: %s",
                audio_path,
                getattr(info, "sample_rate", "unknown"),
                getattr(info, "num_channels", "unknown"),
                bits_per_sample,
            )
    except Exception as meta_err:
        try:
            sf_info = sf.info(audio_path)
            info = sf_info
            bits_per_sample = 16 if getattr(sf_info, "subtype", None) == "PCM_16" else getattr(sf_info, "subtype", None)
            logger.info(
                "Audio prompt metadata (soundfile) - path: %s, sample_rate: %s, channels: %s, subtype: %s",
                audio_path,
                getattr(sf_info, "samplerate", "unknown"),
                getattr(sf_info, "channels", "unknown"),
                getattr(sf_info, "subtype", "unknown"),
            )
        except Exception:
            logger.warning(f"Failed to read audio metadata for {audio_path}: {meta_err}")
            info = None
            bits_per_sample = None

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as load_err:
        logger.warning(f"torchaudio.load failed for {audio_path}: {load_err}. Falling back to soundfile.")
        ffmpeg_decode_path = None
        try:
            data, sample_rate = sf.read(audio_path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
        except Exception as sf_err:
            logger.warning(f"soundfile read failed for {audio_path}: {sf_err}. Attempting ffmpeg decode.")
            try:
                os.makedirs(PROMPT_SANITIZED_DIR, exist_ok=True)
                ffmpeg_decode_path = os.path.join(PROMPT_SANITIZED_DIR, f"decode_{uuid.uuid4().hex}.wav")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-ac",
                    str(PROMPT_EXPECTED_CHANNELS),
                    "-ar",
                    str(target_sr),
                    "-sample_fmt",
                    "s16",
                    ffmpeg_decode_path,
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                data, sample_rate = sf.read(ffmpeg_decode_path, always_2d=True)
                waveform = torch.from_numpy(data.T).float()
            except Exception as ffmpeg_err:
                raise RuntimeError(f"Failed to decode audio prompt {audio_path}: {ffmpeg_err}") from ffmpeg_err
            finally:
                if ffmpeg_decode_path and os.path.exists(ffmpeg_decode_path):
                    try:
                        os.remove(ffmpeg_decode_path)
                    except Exception:
                        pass
    channels = waveform.shape[0]
    min_val = waveform.min().item()
    max_val = waveform.max().item()

    logger.info(
        "Audio prompt stats - sr: %d, channels: %d, min: %.4f, max: %.4f",
        sample_rate,
        channels,
        min_val,
        max_val,
    )

    needs_mono = channels != PROMPT_EXPECTED_CHANNELS
    needs_resample = sample_rate != target_sr
    exceeds_range = min_val < -1.0 or max_val > 1.0

    # Clamp regardless to prevent any stray values
    waveform = torch.clamp(waveform, -1.0, 1.0)

    if needs_mono:
        waveform = waveform.mean(dim=0, keepdim=True)
        logger.info("Audio prompt converted to mono")

    if needs_resample:
        logger.info("Resampling audio prompt from %dHz to %dHz", sample_rate, target_sr)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    info_sample_rate = None
    info_channels = None
    if info is not None:
        info_sample_rate = getattr(info, "sample_rate", getattr(info, "samplerate", None))
        info_channels = getattr(info, "num_channels", getattr(info, "channels", None))

    already_compliant = (
        not needs_mono
        and not needs_resample
        and not exceeds_range
        and info_sample_rate == target_sr
        and info_channels == PROMPT_EXPECTED_CHANNELS
        and bits_per_sample in (16, "PCM_16", "PCM_16LE")
    )

    if already_compliant:
        logger.info("Audio prompt already 24kHz mono PCM16; using original file")
        return audio_path, None

    os.makedirs(PROMPT_SANITIZED_DIR, exist_ok=True)
    sanitized_path = os.path.join(PROMPT_SANITIZED_DIR, f"prompt_{uuid.uuid4().hex}.wav")
    save_waveform_with_soundfile(sanitized_path, waveform, target_sr, subtype="PCM_16")
    logger.info("Audio prompt sanitized and saved to %s", sanitized_path)
    return sanitized_path, sanitized_path


def apply_random_seed(seed_value):
    """Apply deterministic seed across Python, NumPy, and PyTorch with graceful CUDA fallback."""
    try:
        seed_int = int(seed_value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid random seed '{seed_value}' provided; skipping deterministic seeding.")
        return None

    if seed_int < 0:
        logger.warning(f"Random seed {seed_int} is negative; converting to positive value.")
        seed_int = abs(seed_int)

    max_seed = 2 ** 63 - 1
    if seed_int > max_seed:
        logger.warning(f"Random seed {seed_int} exceeds {max_seed}; reducing modulo range.")
        seed_int %= max_seed

    random.seed(seed_int)
    np.random.seed(seed_int)

    try:
        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed_int)
            except Exception as cuda_err:
                logger.warning(f"Failed to seed CUDA RNG (continuing without GPU determinism): {cuda_err}")
    except Exception as torch_err:
        logger.warning(f"Failed to seed PyTorch RNG (continuing without determinism): {torch_err}")

    return seed_int

def normalize_audio(waveform, target_lufs=-23.0, max_peak_db=-1.0):
    """
    Normalize audio to prevent loud distortions and protect speakers.
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        target_lufs (float): Target LUFS (Loudness Units relative to Full Scale)
        max_peak_db (float): Maximum peak level in dB (should be negative)
    
    Returns:
        torch.Tensor: Normalized audio waveform
    """
    logger.info("Applying audio normalization to prevent loud distortions...")
    
    # Ensure waveform is 2D (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Calculate current peak level
    current_peak = torch.max(torch.abs(waveform))
    current_peak_db = 20 * torch.log10(current_peak + 1e-8)  # Add small epsilon to avoid log(0)
    
    logger.info(f"Current peak level: {current_peak_db:.2f} dB")
    
    # Apply hard limiting first to prevent clipping
    peak_limit_linear = 10 ** (max_peak_db / 20)
    if current_peak > peak_limit_linear:
        # Hard limit to prevent speaker damage
        waveform = waveform / current_peak * peak_limit_linear
        logger.info(f"Applied hard limiting to {max_peak_db} dB")
    
    # Apply RMS-based normalization for consistent loudness
    rms = torch.sqrt(torch.mean(waveform ** 2))
    rms_db = 20 * torch.log10(rms + 1e-8)
    
    # Calculate target RMS level (approximate LUFS to RMS conversion)
    target_rms_db = target_lufs + 3.0  # Rough conversion from LUFS to RMS
    
    if rms_db > target_rms_db:
        # Reduce level if too loud
        gain_db = target_rms_db - rms_db
        gain_linear = 10 ** (gain_db / 20)
        waveform = waveform * gain_linear
        logger.info(f"Applied gain reduction: {gain_db:.2f} dB (RMS: {rms_db:.2f} -> {target_rms_db:.2f} dB)")
    
    # Final safety check - ensure we never exceed the peak limit
    final_peak = torch.max(torch.abs(waveform))
    if final_peak > peak_limit_linear:
        waveform = waveform / final_peak * peak_limit_linear
        logger.info("Applied final safety limiting")
    
    final_peak_db = 20 * torch.log10(torch.max(torch.abs(waveform)) + 1e-8)
    final_rms_db = 20 * torch.log10(torch.sqrt(torch.mean(waveform ** 2)) + 1e-8)
    logger.info(f"Final levels - Peak: {final_peak_db:.2f} dB, RMS: {final_rms_db:.2f} dB")
    
    return waveform

def apply_soft_limiting(waveform, threshold_db=-3.0, ratio=10.0):
    """
    Apply soft limiting/compression to prevent harsh clipping.
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        threshold_db (float): Threshold in dB where limiting starts
        ratio (float): Compression ratio (higher = more limiting)
    
    Returns:
        torch.Tensor: Limited audio waveform
    """
    logger.info(f"Applying soft limiting with threshold: {threshold_db} dB, ratio: {ratio}:1")
    
    # Ensure waveform is 2D
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Get the absolute values and signs
    abs_waveform = torch.abs(waveform)
    sign_waveform = torch.sign(waveform)
    
    # Apply soft limiting where signal exceeds threshold
    mask = abs_waveform > threshold_linear
    
    if torch.any(mask):
        # Calculate compressed levels for samples above threshold
        excess_db = 20 * torch.log10(abs_waveform[mask] / threshold_linear + 1e-8)
        compressed_excess_db = excess_db / ratio
        compressed_linear = threshold_linear * (10 ** (compressed_excess_db / 20))
        
        # Replace values above threshold with compressed versions
        abs_waveform[mask] = compressed_linear
        
        # Restore original signs
        waveform = sign_waveform * abs_waveform
        
        logger.info(f"Applied soft limiting to {torch.sum(mask).item()} samples")
    
    return waveform

def convert_audio_to_wav(input_path, sample_rate=24000, output_dir="/app/audio/converted"):
    """
    Convert any audio file to 16-bit mono WAV format with specified sample rate.
    
    Args:
        input_path (str): Path to input audio file
        sample_rate (int): Target sample rate (default: 24000)
        output_dir (str): Directory to save converted file
    
    Returns:
        str: Path to converted WAV file, or None if conversion failed
    """
    logger.info(f"Converting audio file: {input_path} to 16-bit WAV at {sample_rate}Hz")
    start_time = datetime.now()
    
    if not os.path.exists(input_path):
        logger.error(f"Input audio file not found: {input_path}")
        return None
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        input_basename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(input_basename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{name_without_ext}_{sample_rate}hz_{timestamp}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Log input file info
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        logger.info(f"Input file size: {input_size:.2f}MB")
        
        try:
            # Method 1: Try PyTorch/torchaudio conversion (faster)
            logger.info("Attempting PyTorch audio conversion...")
            
            # Load audio with torchaudio
            waveform, orig_sample_rate = torchaudio.load(input_path)
            logger.info(f"Original format - Sample rate: {orig_sample_rate}Hz, Channels: {waveform.shape[0]}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo to mono")
            
            # Resample if needed
            if orig_sample_rate != sample_rate:
                logger.info(f"Resampling from {orig_sample_rate}Hz to {sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
                waveform = resampler(waveform)
            
            # Save as 16-bit WAV
            logger.info(f"Saving converted audio to: {output_path}")
            save_waveform_with_soundfile(output_path, waveform, sample_rate, subtype="PCM_16")
            
            # Verify the conversion
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                logger.info("✓ PyTorch conversion successful")
                conversion_method = "PyTorch"
            else:
                raise RuntimeError("PyTorch conversion produced invalid file")
                
        except Exception as e:
            logger.warning(f"PyTorch conversion failed: {e}, falling back to FFmpeg...")
            
            # Method 2: FFmpeg fallback (more reliable for various formats)
            logger.info("Using FFmpeg for audio conversion...")
            
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # -y to overwrite output file
                "-i", input_path,  # input file
                "-ac", "1",       # mono (1 channel)
                "-ar", str(sample_rate),  # target sample rate
                "-sample_fmt", "s16",  # 16-bit signed integer
                "-f", "wav",      # WAV format
                output_path       # output file
            ]
            
            try:
                logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                logger.info("✓ FFmpeg conversion completed successfully")
                conversion_method = "FFmpeg"
                
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg conversion failed: {e}")
                logger.error(f"FFmpeg stderr: {e.stderr}")
                return None
                
            except FileNotFoundError:
                logger.error("FFmpeg not found! Cannot convert audio file")
                return None
        
        # Verify output file
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✓ Audio conversion completed in {duration:.2f}s using {conversion_method}")
            logger.info(f"Output file: {output_path} ({output_size:.2f}MB)")
            
            # Verify audio format with FFprobe if available
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", output_path
                ]
                probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                probe_data = json.loads(probe_result.stdout)
                if "streams" in probe_data and len(probe_data["streams"]) > 0:
                    audio_stream = probe_data["streams"][0]
                    logger.info(
                        f"Verified format - Channels: {audio_stream.get('channels', 'unknown')}, "
                        f"Sample Rate: {audio_stream.get('sample_rate', 'unknown')}, "
                        f"Bit Depth: {audio_stream.get('bits_per_sample', 'unknown')}"
                    )
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                logger.info("Could not verify audio format with ffprobe")
            
            return output_path
        else:
            logger.error(f"Conversion failed - output file not found: {output_path}")
            return None
            
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Audio conversion error after {duration:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# S3 configuration and helpers
S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX") or os.environ.get("S3_DIR") or "uploads"
S3_REGION = os.environ.get("S3_REGION") or os.environ.get("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
LOG_FILE_PATH = os.environ.get("APP_LOG_PATH", "/app/app.log")
S3_LOG_UPLOAD_ENABLED = os.environ.get("S3_LOG_UPLOAD_ENABLED", "true").lower() in ("true", "1", "yes")
S3_LOG_UPLOAD_INTERVAL_SECONDS = int(os.environ.get("S3_LOG_UPLOAD_INTERVAL_SECONDS", "10"))

_default_log_prefix = None
if S3_PREFIX:
    _default_log_prefix = f"{S3_PREFIX.rstrip('/')}/logs"
S3_LOG_PREFIX = os.environ.get("S3_LOG_PREFIX") or _default_log_prefix or "logs"
S3_LOG_UPLOAD_STOP = threading.Event()
S3_LOG_UPLOAD_THREAD = None

_s3_client = None

def get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        session_kwargs = {}
        client_kwargs = {}
        if S3_REGION:
            session_kwargs["region_name"] = S3_REGION
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client_kwargs.update({
                "aws_access_key_id": AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
            })
        session = boto3.session.Session(**session_kwargs) if session_kwargs else boto3
        _s3_client = session.client("s3", **client_kwargs)
        logger.info("✓ Initialized S3 client")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        return None

def s3_upload_file(local_path, key_prefix=S3_PREFIX):
    if not S3_BUCKET:
        logger.info("S3_BUCKET not set; skipping S3 upload")
        return None
    client = get_s3_client()
    if client is None:
        return None
    try:
        basename = os.path.basename(local_path)
        key_parts = [p for p in [key_prefix, basename] if p]
        s3_key = "/".join([part.strip("/") for part in key_parts])
        # logger.info(f"Uploading {local_path} to s3://{S3_BUCKET}/{s3_key}")
        client.upload_file(local_path, S3_BUCKET, s3_key)
        url = f"s3://{S3_BUCKET}/{s3_key}"
        # Try to construct HTTPS URL if region and public access allow
        https_url = None
        try:
            region = S3_REGION or client.meta.region_name
            if region:
                https_url = f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"
        except Exception:
            pass
        return {"bucket": S3_BUCKET, "key": s3_key, "s3_uri": url, "https_url": https_url}
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected S3 upload error: {e}")
        return {"error": str(e)}

def s3_upload_file_async(local_path, key_prefix=S3_PREFIX, callback=None):
    """
    Upload a file to S3 asynchronously using the thread pool.
    
    Args:
        local_path (str): Path to the local file to upload
        key_prefix (str): S3 key prefix
        callback (callable): Optional callback function to call when upload completes
    
    Returns:
        Future: Future object representing the upload operation
    """
    def upload_with_callback():
        try:
            result = s3_upload_file(local_path, key_prefix)
            if callback:
                callback(result, local_path)
            return result
        except Exception as e:
            logger.error(f"Async S3 upload error for {local_path}: {e}")
            error_result = {"error": str(e)}
            if callback:
                callback(error_result, local_path)
            return error_result
    
    logger.info(f"Submitting async S3 upload for: {local_path}")
    future = S3_UPLOAD_EXECUTOR.submit(upload_with_callback)
    S3_UPLOAD_FUTURES.put(future)
    return future

def cleanup_completed_uploads():
    """Clean up completed upload futures to prevent memory leaks"""
    completed_futures = []
    try:
        # Check completed futures without blocking
        while not S3_UPLOAD_FUTURES.empty():
            try:
                future = S3_UPLOAD_FUTURES.get_nowait()
                if future.done():
                    completed_futures.append(future)
                else:
                    # Put it back if not done
                    S3_UPLOAD_FUTURES.put(future)
                    break
            except queue.Empty:
                break
    except Exception as e:
        logger.warning(f"Error during upload cleanup: {e}")
    
    if completed_futures:
        logger.info(f"Cleaned up {len(completed_futures)} completed upload futures")

def get_s3_upload_status():
    """Get status of S3 uploads"""
    try:
        pending_count = S3_UPLOAD_FUTURES.qsize()
        total_threads = S3_UPLOAD_EXECUTOR._threads
        active_threads = len([t for t in total_threads if t.is_alive()]) if total_threads else 0
        
        return {
            "pending_uploads": pending_count,
            "active_threads": active_threads,
            "max_workers": S3_UPLOAD_EXECUTOR._max_workers,
            "executor_shutdown": S3_UPLOAD_EXECUTOR._shutdown
        }
    except Exception as e:
        logger.warning(f"Error getting S3 upload status: {e}")
        return {"error": str(e)}


# HearsAid API (FastAPI) -----------------------------------------------------
class AudiobookStatusPayload(BaseModel):
    status: str
    detail: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


HEARSAID_API_APP = FastAPI(
    title="HearsAid API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
hearsaid_router = APIRouter(prefix=HEARSAID_API_BASE_PATH, tags=["hearsaid"])

HEARSAID_SERVER: Optional[uvicorn.Server] = None
HEARSAID_SERVER_THREAD: Optional[threading.Thread] = None


def validate_upload_filename(filename: Optional[str]) -> str:
    if filename is None:
        raise ValueError("Filename is required")

    raw_name = filename
    if raw_name.strip() == "":
        raise ValueError("Filename cannot be empty")

    if "\x00" in raw_name:
        raise ValueError("Filename contains null bytes")

    if os.path.isabs(raw_name):
        raise ValueError("Absolute paths are not allowed in filenames")

    if os.path.basename(raw_name) != raw_name or "\\" in raw_name:
        raise ValueError("Filename must not contain path separators")

    if raw_name in {".", ".."}:
        raise ValueError("Reserved filenames are not allowed")

    if len(raw_name) > 255:
        raise ValueError("Filename is too long")

    return raw_name


def safe_join(base_dir: str, *paths: str) -> str:
    candidate = os.path.realpath(os.path.join(base_dir, *paths))
    base_real = os.path.realpath(base_dir)
    if not candidate.startswith(base_real.rstrip(os.sep) + os.sep) and candidate != base_real:
        raise ValueError("Path traversal detected in archive member")
    return candidate


def safe_extract_zip(zip_path: str, target_dir: str) -> List[str]:
    extracted_files: List[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            normalized_name = os.path.normpath(member.filename)
            if normalized_name.startswith("..") or normalized_name.startswith("/"):
                raise ValueError(f"Unsafe path inside zip: {member.filename}")
            if normalized_name.startswith("__MACOSX"):
                continue  # skip macOS metadata folders
            destination_path = safe_join(target_dir, normalized_name)
            if member.is_dir():
                os.makedirs(destination_path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            with archive.open(member) as src, open(destination_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(destination_path)
    return extracted_files


async def persist_uploaded_file(upload: UploadFile, destination: str) -> int:
    total_bytes = 0
    with open(destination, "wb") as out_file:
        while True:
            chunk = await upload.read(HEARSAID_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            out_file.write(chunk)
            total_bytes += len(chunk)
    await upload.close()
    return total_bytes


def schedule_pod_shutdown(reason: str):
    def _shutdown_task():
        try:
            perform_pod_shutdown(shutdown_reason=reason, logger=logger)
        except Exception as shutdown_err:
            logger.error(f"Pod shutdown routine raised: {shutdown_err}")
        finally:
            logger.info("Exiting pod process after HearsAid shutdown request")
            os._exit(0)

    threading.Thread(target=_shutdown_task, name="hearsaid-shutdown", daemon=True).start()


@hearsaid_router.post("/audiobook-status")
async def update_audiobook_status(
    payload: AudiobookStatusPayload,
    background_tasks: BackgroundTasks,
):
    status_value = (payload.status or "").strip().lower()
    if status_value == "processing":
        logger.info(
            "Audiobook processing heartbeat received%s",
            f": {payload.detail}" if payload.detail else "",
        )
        return {
            "status": "acknowledged",
            "action": "heartbeat",
            "detail": payload.detail,
        }
    if status_value == "exited":
        logger.info("Audiobook processor reported exit; scheduling pod shutdown")
        background_tasks.add_task(schedule_pod_shutdown, "audiobook_status_exit")
        return {
            "status": "acknowledged",
            "action": "shutdown",
            "detail": payload.detail,
        }

    raise HTTPException(status_code=400, detail="Unsupported status. Use 'processing' or 'exited'.")


@hearsaid_router.post("/audio-upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")

    try:
        original_name = validate_upload_filename(file.filename)
    except ValueError as validation_err:
        raise HTTPException(status_code=400, detail=str(validation_err)) from validation_err

    extension = os.path.splitext(original_name)[1].lower()
    if extension not in HEARSAID_ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: wav, mp3, zip.",
        )

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    destination_path = os.path.join(AUDIO_OUTPUT_DIR, original_name)
    logger.info(f"Receiving upload for {original_name} -> {destination_path}")

    bytes_written = await persist_uploaded_file(file, destination_path)

    response: Dict[str, Any] = {
        "filename": original_name,
        "saved_path": destination_path,
        "bytes_written": bytes_written,
    }

    if extension == ".zip":
        if not zipfile.is_zipfile(destination_path):
            logger.error("Uploaded file %s is not a valid zip archive", destination_path)
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive")
        try:
            extracted_files = safe_extract_zip(destination_path, AUDIO_OUTPUT_DIR)
            response["unzipped_files"] = extracted_files
            logger.info(
                "Stored zip %s and extracted %d entries", destination_path, len(extracted_files)
            )
        except Exception as zip_err:
            logger.error(f"Failed to extract {destination_path}: {zip_err}")
            raise HTTPException(status_code=400, detail=f"Zip extraction failed: {zip_err}")
    else:
        logger.info(
            "Stored audio upload %s (%s) - %.2f MB",
            destination_path,
            extension,
            bytes_written / (1024 * 1024) if bytes_written else 0,
        )

    return response


HEARSAID_API_APP.include_router(hearsaid_router)


def start_hearsaid_api_server():
    global HEARSAID_SERVER, HEARSAID_SERVER_THREAD
    if HEARSAID_SERVER_THREAD and HEARSAID_SERVER_THREAD.is_alive():
        return HEARSAID_SERVER_THREAD

    config = uvicorn.Config(
        HEARSAID_API_APP,
        host=HEARSAID_API_HOST,
        port=HEARSAID_API_PORT,
        log_level="info",
    )
    HEARSAID_SERVER = uvicorn.Server(config)

    def _run_server():
        logger.info(
            "Starting HearsAid API on %s:%s with prefix %s",
            HEARSAID_API_HOST,
            HEARSAID_API_PORT,
            HEARSAID_API_BASE_PATH,
        )
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            HEARSAID_SERVER.run()
        except Exception as server_err:
            logger.error(f"HearsAid API server crashed: {server_err}")

    HEARSAID_SERVER_THREAD = threading.Thread(
        target=_run_server,
        name="hearsaid-api-server",
        daemon=True,
    )
    HEARSAID_SERVER_THREAD.start()
    return HEARSAID_SERVER_THREAD


def stop_hearsaid_api_server():
    if HEARSAID_SERVER is not None:
        HEARSAID_SERVER.should_exit = True
    if HEARSAID_SERVER_THREAD is not None and HEARSAID_SERVER_THREAD.is_alive():
        HEARSAID_SERVER_THREAD.join(timeout=5)


def start_periodic_log_upload(log_path=LOG_FILE_PATH):
    """Launch a daemon thread that ships the application log to S3 on an interval."""
    global S3_LOG_UPLOAD_THREAD

    if not S3_LOG_UPLOAD_ENABLED:
        logger.info("Periodic log upload disabled by configuration")
        return

    if not S3_BUCKET:
        logger.info("S3_BUCKET not configured; periodic log upload disabled")
        return

    if S3_LOG_UPLOAD_THREAD and S3_LOG_UPLOAD_THREAD.is_alive():
        return

    def upload_loop():
        logger.info(
            f"Periodic log upload thread started (interval: {S3_LOG_UPLOAD_INTERVAL_SECONDS}s, key prefix: {S3_LOG_PREFIX})"
        )
        while not S3_LOG_UPLOAD_STOP.is_set():
            try:
                if os.path.exists(log_path):
                    result = s3_upload_file(log_path, key_prefix=S3_LOG_PREFIX)
                    if isinstance(result, dict) and result.get("error"):
                        logger.warning(f"Periodic log upload failed: {result['error']}")
                else:
                    logger.warning(f"Periodic log upload skipped; log file missing at {log_path}")
            except Exception as upload_err:
                logger.warning(f"Periodic log upload raised exception: {upload_err}")

            if S3_LOG_UPLOAD_STOP.wait(S3_LOG_UPLOAD_INTERVAL_SECONDS):
                break

        logger.info("Periodic log upload thread stopping")

    S3_LOG_UPLOAD_THREAD = threading.Thread(target=upload_loop, name="log-uploader", daemon=True)
    S3_LOG_UPLOAD_THREAD.start()


start_periodic_log_upload()

def transcribe_whisper(audio_file, sample_rate=24000):
    logger.info(f"Whisper transcription started for file: {audio_file}")
    start_time = datetime.now()
    
    try:
        # Check if Whisper model is available
        if whisper_model is None:
            logger.error("Whisper model is not available (failed to load)")
            return {"error": "Whisper model is not available. Please use VOSK transcription instead."}
        
        # Log file info
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
            logger.info(f"Original audio file size: {file_size:.2f}MB")
        else:
            logger.error(f"Audio file not found: {audio_file}")
            return {"error": "Audio file not found"}
        
        # Convert audio to standard format
        logger.info(f"Converting audio to 16-bit WAV at {sample_rate}Hz...")
        converted_file = convert_audio_to_wav(audio_file, sample_rate)
        
        if not converted_file:
            logger.error("Audio conversion failed")
            return {"error": "Audio conversion failed"}
        
        logger.info(f"Using converted file: {converted_file}")
        
        logger.info("Starting Whisper transcription...")
        result = whisper_model.transcribe(converted_file, word_timestamps=False, beam_size=1)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Whisper transcription completed in {duration:.2f}s")
        logger.info(f"Transcribed text length: {len(result.get('text', ''))}")
        
        # Add conversion info to result
        result["audio_conversion"] = {
            "original_file": audio_file,
            "converted_file": converted_file,
            "target_sample_rate": sample_rate,
            "format": "16-bit mono WAV"
        }
        
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Whisper transcription error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def transcribe_whisper_filepath(file_path, sample_rate=24000):
    """Whisper transcription function that accepts server-side file paths"""
    logger.info(f"Whisper filepath transcription started for: {file_path}")
    
    # Check if Whisper model is available
    if whisper_model is None:
        logger.error("Whisper model is not available (failed to load)")
        return {"error": "Whisper model is not available. Please use VOSK transcription instead."}
    
    # Check if file exists on server
    if not os.path.exists(file_path):
        error_msg = f"File not found on server: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Use the existing transcribe_whisper function
    return transcribe_whisper(file_path, sample_rate)

def transcribe_vosk(audio_file, sample_rate=24000):
    logger.info(f"VOSK transcription started for file: {audio_file} with sample rate: {sample_rate}")
    start_time = datetime.now()
    
    if vosk_model is None:
        logger.error("VOSK model not available. Transcription disabled.")
        return {"error": "VOSK model not available"}
    
    try:
        # Log file info
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
            logger.info(f"Original audio file size: {file_size:.2f}MB")
        else:
            logger.error(f"Audio file not found: {audio_file}")
            return {"error": "Audio file not found"}
        
        # Convert audio to standard format
        logger.info(f"Converting audio to 16-bit WAV at {sample_rate}Hz...")
        converted_file = convert_audio_to_wav(audio_file, sample_rate)
        
        if not converted_file:
            logger.error("Audio conversion failed")
            return {"error": "Audio conversion failed"}
        
        logger.info(f"Using converted file: {converted_file}")
        
        # Initialize the recognizer with the model
        logger.info(f"Initializing VOSK recognizer with sample rate: {sample_rate}...")
        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        recognizer.SetWords(True)
        
        # Open the converted audio file
        logger.info("Processing audio chunks...")
        chunks_processed = 0
        with open(converted_file, "rb") as audio:
            while True:
                # Read a chunk of the audio file
                data = audio.read(4000)
                if len(data) == 0:
                    break
                # Recognize the speech in the chunk
                recognizer.AcceptWaveform(data)
                chunks_processed += 1

        logger.info(f"Processed {chunks_processed} audio chunks")
        result = recognizer.FinalResult()
        result_dict = json.loads(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ VOSK transcription completed in {duration:.2f}s")
        logger.info(f"Transcribed text: {result_dict.get('text', 'No text')}")
        
        # Add conversion info to result
        result_dict["audio_conversion"] = {
            "original_file": audio_file,
            "converted_file": converted_file,
            "target_sample_rate": sample_rate,
            "format": "16-bit mono WAV"
        }
        
        return result_dict
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"VOSK transcription error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def transcribe_vosk_filepath(file_path, sample_rate=24000):
    """VOSK transcription function that accepts server-side file paths"""
    logger.info(f"VOSK filepath transcription started for: {file_path}")
    
    # Check if file exists on server
    if not os.path.exists(file_path):
        error_msg = f"File not found on server: {file_path}"
        logger.error(error_msg)
        return {"error": error_msg}
    
    # Use the existing transcribe_vosk function
    return transcribe_vosk(file_path, sample_rate)

def chatterbox_clone(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None, output_filename=None, skip_conversion=None):
    logger.info(f"Chatterbox TTS started for text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    start_time = datetime.now()
    
    try:
        # Log parameters
        logger.info(f"Parameters - exaggeration: {exaggeration}, cfg_weight: {cfg_weight}, temperature: {temperature}, seed: {random_seed}, custom_filename: {output_filename}")
        
        # Create output directory if it doesn't exist
        output_dir = AUDIO_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")
        
        # Determine output filename based on priority: custom filename > audio prompt filename > timestamp
        if output_filename:
            # User provided custom filename
            name_without_ext = os.path.splitext(output_filename)[0]
            final_output_filename = f"{name_without_ext}.wav"
            logger.info(f"Using custom filename: {final_output_filename}")
        elif audio_prompt and os.path.exists(audio_prompt):
            prompt_size = os.path.getsize(audio_prompt) / (1024 * 1024)  # MB
            logger.info(f"Using audio prompt: {audio_prompt} ({prompt_size:.2f}MB)")
            
            # Extract filename from audio prompt path and use it for output
            input_filename = os.path.basename(audio_prompt)
            # Keep the same name but ensure .wav extension
            name_without_ext = os.path.splitext(input_filename)[0]
            final_output_filename = f"{name_without_ext}.wav"
            logger.info(f"Using audio prompt filename: {final_output_filename}")
        else:
            # If no custom filename or audio prompt, create a generic filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_filename = f"chatterbox_output_{timestamp}.wav"
            logger.info("No custom filename or audio prompt provided, using timestamp-based filename")
        
        output_path = os.path.join(output_dir, final_output_filename)
        logger.info(f"Output will be saved as: {output_path}")
        
        # Prepare generation parameters
        generation_params = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature
        }

        # Serialize GPU-bound chatterbox generation to avoid device-side asserts from concurrent kernels
        lock_acquired = CHATTERBOX_GENERATION_LOCK.acquire(blocking=False)
        if not lock_acquired:
            logger.info("Chatterbox GPU is busy; waiting for the current generation to finish...")
            CHATTERBOX_GENERATION_LOCK.acquire()
            lock_acquired = True

        prompt_cleanup_path = None
        try:
            if random_seed is not None:
                applied_seed = apply_random_seed(random_seed)
                if applied_seed is not None:
                    logger.info(f"Set random seed: {applied_seed}")
                else:
                    logger.info("Random seed could not be applied; proceeding with non-deterministic RNG state.")

            logger.info("Generating audio...")

            sanitized_prompt_path = None
            if audio_prompt and os.path.exists(audio_prompt):
                try:
                    sanitized_prompt_path, prompt_cleanup_path = ensure_prompt_audio_format(audio_prompt)
                    if prompt_cleanup_path:
                        logger.info(f"Using sanitized audio prompt copy: {sanitized_prompt_path}")
                    else:
                        logger.info("Audio prompt passed validation without changes")
                except Exception as prompt_err:
                    logger.error(f"Audio prompt validation failed ({audio_prompt}): {prompt_err}")
                    raise

            if sanitized_prompt_path:
                wav = chatterbox_model.generate(text, audio_prompt_path=sanitized_prompt_path, **generation_params)
            else:
                wav = chatterbox_model.generate(text, **generation_params)
        finally:
            if prompt_cleanup_path and prompt_cleanup_path != audio_prompt:
                try:
                    os.remove(prompt_cleanup_path)
                    logger.info(f"Removed temporary sanitized prompt: {prompt_cleanup_path}")
                except FileNotFoundError:
                    pass
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete temporary prompt {prompt_cleanup_path}: {cleanup_err}")
            if lock_acquired:
                CHATTERBOX_GENERATION_LOCK.release()
        
        # Apply audio normalization to prevent loud distortions and protect speakers
        if AUDIO_NORMALIZATION_ENABLED:
            logger.info("Applying audio normalization and safety limiting...")
            wav = apply_soft_limiting(wav, threshold_db=AUDIO_LIMITER_THRESHOLD_DB, ratio=AUDIO_LIMITER_RATIO)  # Soft limiting first
            wav = normalize_audio(wav, target_lufs=AUDIO_TARGET_LUFS, max_peak_db=AUDIO_MAX_PEAK_DB)  # Then normalize
        else:
            logger.info("Audio normalization disabled by configuration")
        
        # Use skip_conversion parameter, defaulting to environment variable
        if skip_conversion is None:
            skip_conversion = SKIP_AUDIO_CONVERSION
        
        conversion_method = "None (skipped)"
        
        if skip_conversion:
            logger.info("Skipping audio conversion (using original format)...")
            # Save directly with original format (faster)
            try:
                logger.info(f"Saving audio directly to: {output_path} (original format)")
                save_waveform_with_soundfile(output_path, wav, chatterbox_model.sr, subtype="FLOAT")
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    logger.info("✓ Direct save successful (no conversion)")
                    conversion_method = "Direct (no conversion)"
                else:
                    raise RuntimeError("Direct save failed")
            except Exception as e:
                logger.error(f"Direct save failed: {e}")
                # Fallback to conversion if direct save fails
                skip_conversion = False
                logger.info("Falling back to audio conversion...")
        
        if not skip_conversion:
            # Try direct encoding first (faster), fallback to FFmpeg if needed
            logger.info("Converting audio to standard format (mono, 24kHz, 16-bit)...")
            
            try:
                # Method 1: Direct PyTorch conversion (faster)
                logger.info("Attempting direct PyTorch conversion...")
                
                # Convert to mono if stereo
                if wav.shape[0] > 1:
                    wav = torch.mean(wav, dim=0, keepdim=True)
                    logger.info("Converted stereo to mono")
                
                # Resample to 24kHz if needed
                original_sr = chatterbox_model.sr
                target_sr = 24000
                if original_sr != target_sr:
                    logger.info(f"Resampling from {original_sr}Hz to {target_sr}Hz")
                    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                    wav = resampler(wav)
                
                # Save directly with target format
                logger.info(f"Saving audio directly to: {output_path}")
                save_waveform_with_soundfile(output_path, wav, target_sr, subtype="PCM_16")
                
                # Verify the file was created successfully
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # At least 1KB
                    logger.info("✓ Direct PyTorch conversion successful")
                    conversion_method = "PyTorch (direct)"
                else:
                    raise RuntimeError("Direct conversion produced invalid file")
                    
            except Exception as e:
                logger.warning(f"Direct conversion failed: {e}, falling back to FFmpeg...")
                conversion_method = "FFmpeg (fallback)"
                
                # Method 2: FFmpeg fallback (more reliable)
                logger.info("Saving to temporary file for FFmpeg conversion...")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                # Close file before writing with soundfile
                save_waveform_with_soundfile(temp_path, wav, chatterbox_model.sr, subtype="PCM_16")
                
                logger.info(f"Converting with FFmpeg...")
                
                # Use FFmpeg to convert to desired format
                ffmpeg_cmd = [
                    "ffmpeg", "-y",  # -y to overwrite output file
                    "-i", temp_path,  # input file
                    "-ac", "1",       # mono (1 channel)
                    "-ar", "24000",   # 24kHz sample rate
                    "-sample_fmt", "s16",  # 16-bit signed integer
                    "-f", "wav",      # WAV format
                    output_path       # output file
                ]
                
                try:
                    logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                    logger.info("✓ FFmpeg conversion completed successfully")
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    logger.info("Temporary file cleaned up")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e}")
                    logger.error(f"FFmpeg stderr: {e.stderr}")
                    
                    # Last resort: move temp file to final location
                    logger.info("Using original audio format (no conversion)")
                    os.rename(temp_path, output_path)
                    conversion_method = "Original (no conversion)"
                    
                except FileNotFoundError:
                    logger.error("FFmpeg not found! Using original audio format")
                    os.rename(temp_path, output_path)
                    conversion_method = "Original (FFmpeg not found)"
        
        logger.info(f"Audio saved as: {output_path} (method: {conversion_method})")
        
        # Log output info
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Chatterbox TTS completed in {duration:.2f}s")
            logger.info(f"Generated audio file: {output_path} ({output_size:.2f}MB)")
            
            # Verify audio format with FFprobe if available (restored)
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", output_path
                ]
                probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                probe_data = json.loads(probe_result.stdout)
                if "streams" in probe_data and len(probe_data["streams"]) > 0:
                    audio_stream = probe_data["streams"][0]
                    logger.info(
                        f"Audio format verification - Channels: {audio_stream.get('channels', 'unknown')}, "
                        f"Sample Rate: {audio_stream.get('sample_rate', 'unknown')}, "
                        f"Bit Depth: {audio_stream.get('bits_per_sample', 'unknown')}"
                    )
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                logger.info("Could not verify audio format with ffprobe")
            
            # Upload to S3 asynchronously if configured
            s3_future = None
            s3_info = None
            
            def s3_upload_callback(result, file_path):
                """Callback function for async S3 upload completion"""
                if "error" in result:
                    logger.error(f"Async S3 upload failed for {file_path}: {result['error']}")
                else:
                    logger.info(f"Async S3 upload completed for {file_path}: {result}")
            
            if S3_BUCKET:
                logger.info("Starting async S3 upload...")
                s3_future = s3_upload_file_async(output_path, callback=s3_upload_callback)
                s3_info = {"status": "uploading", "future_id": id(s3_future)}
                
                # Clean up any completed uploads to prevent memory leaks
                cleanup_completed_uploads()
            else:
                logger.info("S3_BUCKET not configured; skipping S3 upload")
                s3_info = {"status": "skipped", "reason": "S3_BUCKET not configured"}
            # Determine audio format description based on conversion
            if skip_conversion:
                audio_format_desc = f"Original format (sample rate: {chatterbox_model.sr}Hz)"
            else:
                audio_format_desc = "mono WAV, 24kHz, 16-bit"
            
            # Return detailed result for API
            result = {
                "audio_file": output_path,
                "output_path": output_path,
                "filename": final_output_filename,
                "file_size_mb": round(output_size, 2),
                "generation_time_seconds": round(duration, 2),
                "audio_format": audio_format_desc,
                "conversion_method": conversion_method,
                "skip_conversion": skip_conversion,
                "parameters": {
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "temperature": temperature,
                    "random_seed": random_seed,
                    "custom_filename": output_filename,
                    "skip_conversion": skip_conversion
                },
                "text_length": len(text),
                "used_audio_prompt": bool(audio_prompt and os.path.exists(audio_prompt)),
                "s3": s3_info,
            }
            logger.info(f"API result: {result}")
            return result
        else:
            error_msg = f"Generated file not found at {output_path}"
            logger.error(error_msg)
            return {"error": error_msg, "output_path": output_path}
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Chatterbox cloning error after {duration:.2f}s: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def chatterbox_clone_gradio(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None, output_filename=None, skip_conversion=None):
    """Wrapper function for Gradio interface that returns just the audio file path"""
    result = chatterbox_clone(text, audio_prompt, exaggeration, cfg_weight, temperature, random_seed, output_filename, skip_conversion)
    
    if isinstance(result, dict):
        if "error" in result:
            return None, result  # Return None for audio, error dict for JSON
        else:
            return result["audio_file"], result  # Return audio file path and full result
    else:
        return result, {"error": "Unexpected result format"}  # Fallback

# Gradio Interface with custom endpoint names and new parameters
logger.info("Setting up Gradio interfaces...")

try:
    logger.info("Creating Whisper interface...")
    whisper_iface = gr.Interface(
        fn=transcribe_whisper,
        inputs=[
            gr.Audio(type="filepath", label="Upload audio for Whisper transcription"),
            gr.Number(label="Sample Rate", value=24000, precision=0, info="Target sample rate for audio conversion (Hz)")
        ],
        outputs=gr.JSON(label="Whisper Result"),
        title="OpenAI Whisper Turbo Transcription",
        api_name="whisper"
    )
    logger.info("✓ Whisper interface created")

    logger.info("Creating Whisper filepath interface...")
    whisper_filepath_iface = gr.Interface(
        fn=transcribe_whisper_filepath,
        inputs=[
            gr.Textbox(label="Server File Path", placeholder="/app/audio/output/filename.wav", info="Path to audio file on server"),
            gr.Number(label="Sample Rate", value=24000, precision=0, info="Target sample rate for audio conversion (Hz)")
        ],
        outputs=gr.JSON(label="Whisper Result"),
        title="OpenAI Whisper Turbo (Server Files)",
        api_name="whisper_filepath"
    )
    logger.info("✓ Whisper filepath interface created")

    logger.info("Creating VOSK interface...")
    vosk_iface = gr.Interface(
        fn=transcribe_vosk,
        inputs=[
            gr.Audio(type="filepath", label="Upload audio for VOSK transcription"),
            gr.Number(label="Sample Rate", value=24000, precision=0, info="Target sample rate for audio conversion (Hz)")
        ],
        outputs=gr.JSON(label="VOSK Result"),
        title="VOSK Transcription",
        api_name="vosk"
    )
    logger.info("✓ VOSK interface created")

    logger.info("Creating VOSK filepath interface...")
    vosk_filepath_iface = gr.Interface(
        fn=transcribe_vosk_filepath,
        inputs=[
            gr.Textbox(label="Server File Path", placeholder="/app/audio/output/filename.wav", info="Path to audio file on server"),
            gr.Number(label="Sample Rate", value=24000, precision=0, info="Target sample rate for audio conversion (Hz)")
        ],
        outputs=gr.JSON(label="VOSK Result"),
        title="VOSK Transcription (Server Files)",
        api_name="vosk_filepath"
    )
    logger.info("✓ VOSK filepath interface created")

    logger.info("Creating Chatterbox interface...")
    chatterbox_iface = gr.Interface(
        fn=chatterbox_clone_gradio,
        inputs=[
            gr.Textbox(label="Text to clone"),
            gr.Audio(type="filepath", label="Reference audio (optional for voice cloning)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="Exaggeration (emotion intensity)"),
            gr.Slider(minimum=0, maximum=1, value=0.5, label="CFG Weight (pacing control)"),
            gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature"),
            gr.Number(label="Random Seed", value=None, precision=0),
            gr.Textbox(label="Output Filename (optional)", placeholder="e.g., my_voice_clone", info="Will be saved as .wav file. If empty, uses audio prompt filename or timestamp.")
        ],
        outputs=[
            gr.Audio(type="filepath", label="Generated Audio"),
            gr.JSON(label="Generation Details")
        ],
        title="Resemble.AI Chatterbox Voice Cloning",
        api_name="chatterbox"
    )
    logger.info("✓ Chatterbox interface created")

    logger.info("Creating tabbed interface...")
    app = gr.TabbedInterface([whisper_iface, whisper_filepath_iface, vosk_iface, vosk_filepath_iface, chatterbox_iface], 
                           ["Whisper", "Whisper Files", "VOSK", "VOSK Files", "Chatterbox"])
    logger.info("✓ All Gradio interfaces created successfully")

except Exception as e:
    logger.error(f"✗ Failed to create Gradio interfaces: {e}")
    logger.error(traceback.format_exc())
    raise

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("Starting Gradio application...")
        logger.info(f"Server configuration: 0.0.0.0:7860")
        logger.info("="*50)

        start_hearsaid_api_server()
        
        # Add signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            stop_hearsaid_api_server()
            
            # Shutdown S3 upload executor
            logger.info("Shutting down S3 upload executor...")
            try:
                S3_UPLOAD_EXECUTOR.shutdown(wait=True, timeout=30)
                logger.info("✓ S3 upload executor shutdown completed")
            except Exception as e:
                logger.warning(f"Error shutting down S3 upload executor: {e}")
            
            S3_LOG_UPLOAD_STOP.set()
            perform_pod_shutdown(shutdown_reason=f"signal:{signum}", logger=logger)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Launch the app with detailed logging
        logger.info("Launching Gradio application...")
        app.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error during application startup: {e}")
        logger.critical(traceback.format_exc())
        
        # Try to clean up resources
        try:
            stop_hearsaid_api_server()
            # Shutdown S3 upload executor
            logger.info("Shutting down S3 upload executor...")
            try:
                S3_UPLOAD_EXECUTOR.shutdown(wait=True, timeout=30)
                logger.info("✓ S3 upload executor shutdown completed")
            except Exception as e:
                logger.warning(f"Error shutting down S3 upload executor: {e}")
            
            perform_pod_shutdown(shutdown_reason="startup_exception", logger=logger)
        except:
            pass
            
        sys.exit(1)
    finally:
        stop_hearsaid_api_server()
        S3_LOG_UPLOAD_STOP.set()
        if S3_LOG_UPLOAD_THREAD and S3_LOG_UPLOAD_THREAD.is_alive():
            logger.info("Stopping periodic log upload thread...")
            try:
                S3_LOG_UPLOAD_THREAD.join(timeout=5)
            except Exception as join_err:
                logger.warning(f"Error stopping log upload thread: {join_err}")

        # Shutdown S3 upload executor
        logger.info("Shutting down S3 upload executor...")
        try:
            S3_UPLOAD_EXECUTOR.shutdown(wait=True, timeout=30)
            logger.info("✓ S3 upload executor shutdown completed")
        except Exception as e:
            logger.warning(f"Error shutting down S3 upload executor: {e}")
        
        # Upload logs to S3 on shutdown
        try:
            perform_pod_shutdown(shutdown_reason="finally", logger=logger)
        except Exception as e:
            logger.error(f"Failed during shutdown: {e}")
        logger.info("Application terminating...")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*50)
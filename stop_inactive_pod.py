import os
import time
import re
import subprocess
import logging
import psutil  # Added for process checking
import shutil
import signal
from datetime import datetime
import glob
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from pod_shutdown import perform_pod_shutdown

# Configuration
INITIAL_DELAY = int(os.environ.get("AUTOSHUTOFF_INITIAL_DELAY", "600"))  # 10 minutes to allow boot
CHECK_INTERVAL = int(os.environ.get("AUTOSHUTOFF_CHECK_INTERVAL", "60"))  # Check every 60 seconds
INACTIVITY_THRESHOLD = int(os.environ.get("AUTOSHUTOFF_INACTIVITY_THRESHOLD", "600"))  # 10 minutes until pod stops
LOG_FILE = "/app/app.log"
MONITOR_LOG_FILE = "/app/inactivity_monitor.log"

# S3 configuration (reuses same env vars as app.py)
S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX") or os.environ.get("S3_DIR") or "uploads"
S3_REGION = os.environ.get("S3_REGION") or os.environ.get("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")

# Set up logging for the monitor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MONITOR_LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)
ACTIVITY_PATTERNS = [
    r"Starting ChatterVosker Application",
    r"Loading.*model",
    r"Checking VOSK model",
    r"Initializing models",
    r"Gradio.*interface",
    r"Whisper transcription started",
    r"VOSK transcription started",
    r"Chatterbox TTS started",
    r"POST /",
    r"API result:",
    r"generation_time_seconds",
    r"audio_file.*\.wav",
    r"Application.*started",
]

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
        logging.info("✓ Inactivity monitor initialized S3 client")
        return _s3_client
    except Exception as e:
        logging.error(f"Failed to initialize S3 client in monitor: {e}")
        return None

def s3_upload_file(local_path, key_prefix=S3_PREFIX):
    if not S3_BUCKET:
        logging.info("S3_BUCKET not set; skipping S3 upload from monitor")
        return None
    client = get_s3_client()
    if client is None:
        return None
    try:
        basename = os.path.basename(local_path)
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        key_parts = [p for p in [key_prefix, date_prefix, basename] if p]
        s3_key = "/".join([part.strip("/") for part in key_parts])
        logging.info(f"Uploading log {local_path} to s3://{S3_BUCKET}/{s3_key}")
        client.upload_file(local_path, S3_BUCKET, s3_key)
        return {"bucket": S3_BUCKET, "key": s3_key}
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        logging.error(f"S3 upload failed (monitor): {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected S3 upload error (monitor): {e}")
        return {"error": str(e)}

def upload_logs_to_s3():
    try:
        if not S3_BUCKET:
            logging.info("S3 bucket not configured; skipping log upload")
            return []
        logs = set()
        # Known logs
        for p in ["/app/app.log", "/app/inactivity_monitor.log", "/tmp/vosk_download.log"]:
            if os.path.exists(p):
                logs.add(p)
        # Any .log files in /app
        for p in glob.glob("/app/*.log"):
            if os.path.exists(p):
                logs.add(p)
        if not logs:
            logging.info("No log files found to upload")
            return []
        results = []
        for path in sorted(logs):
            info = s3_upload_file(path, key_prefix=f"{S3_PREFIX}/logs")
            results.append({"file": path, "result": info})
        logging.info(f"Uploaded {len(results)} log files to S3")
        return results
    except Exception as e:
        logging.error(f"Failed to upload logs to S3: {e}")
        return []

def has_recent_activity():
    try:
        # Check if log file exists
        if not os.path.exists(LOG_FILE):
            logger.info(f"Log file {LOG_FILE} not found - assuming active during startup")
            return True  # Assume active if no log file yet (e.g., during boot)

        # Check file modification time
        last_modified = os.path.getmtime(LOG_FILE)
        current_time = time.time()
        time_since_modified = current_time - last_modified
        
        logger.debug(f"Log file last modified {time_since_modified:.1f} seconds ago")
        
        # Consider recent if log file changed within the check interval (more strict than threshold)
        if time_since_modified < max(CHECK_INTERVAL * 2, 30):
            logger.debug("Recent file modification detected; treating as active")
            return True

        # Check recent log lines for activity patterns
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()[-300:]  # Check last 300 lines for efficiency
            recent_lines = []
            
            # Look for lines with timestamps within the threshold
            for line in lines:
                # Extract timestamp from log line (assuming format: YYYY-MM-DD HH:MM:SS,mmm)
                if len(line) > 23:  # Minimum length for timestamp
                    timestamp_str = line[:23]
                    try:
                        # Parse the timestamp - handle milliseconds by splitting on comma
                        if ',' in timestamp_str:
                            datetime_part = timestamp_str.split(',')[0]
                            log_time = time.strptime(datetime_part, "%Y-%m-%d %H:%M:%S")
                        else:
                            log_time = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        log_timestamp = time.mktime(log_time)
                        
                        # Check if this log entry is within our threshold
                        if current_time - log_timestamp < INACTIVITY_THRESHOLD:
                            recent_lines.append(line)
                            # Also check for activity patterns in recent lines
                            for pattern in ACTIVITY_PATTERNS:
                                if re.search(pattern, line, re.IGNORECASE):
                                    logger.info(f"Found recent activity pattern: {pattern}")
                                    logger.debug(f"In log line: {line.strip()}")
                                    return True
                    except ValueError as e:
                        # Skip lines that don't have proper timestamp format
                        logger.debug(f"Could not parse timestamp from: {timestamp_str} - {e}")
                        continue
            
            if recent_lines:
                logger.info(f"Found {len(recent_lines)} recent log entries, but no activity patterns matched")
                # Show the most recent few lines for debugging
                for line in recent_lines[-3:]:
                    logger.debug(f"Recent log: {line.strip()}")
            else:
                logger.info("No recent log entries found within threshold")
                
        return False
    except Exception as e:
        logger.error(f"Error checking log activity: {e}")
        return True  # Assume active if error to avoid premature shutdown

def stop_pod():
    pod_id = os.environ.get("RUNPOD_POD_ID")
    logger.warning("Inactivity threshold reached; initiating shutdown")
    logger.info("Recent log contents:")
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()[-10:]
                for line in lines:
                    logger.info(f"Recent log: {line.strip()}")
    except Exception as e:
        logger.error(f"Error reading log file: {e}")

    # Upload logs to S3 before stopping
    try:
        upload_results = upload_logs_to_s3()
        logger.info(f"S3 log upload results: {upload_results}")
    except Exception as e:
        logger.error(f"Error uploading logs to S3: {e}")

    # Centralized shutdown (idempotent)
    try:
        perform_pod_shutdown(shutdown_reason="inactivity_monitor", logger=logger)
    except Exception as e:
        logger.warning(f"perform_pod_shutdown failed: {e}")

    # Prefer runpodctl if present, otherwise signal PID 1 to trigger container shutdown
    try:
        if shutil.which("runpodctl") and pod_id:
            logger.info(f"Using runpodctl to stop pod {pod_id}")
            subprocess.run(["runpodctl", "stop", "pod", pod_id], check=False)
        else:
            logger.warning("runpodctl not available or RUNPOD_POD_ID missing; sending SIGTERM to PID 1")
            os.kill(1, signal.SIGTERM)
    except Exception as e:
        logger.error(f"Failed to stop pod/container gracefully: {e}")

def main():
    logger.info("="*50)
    logger.info("Starting Inactivity Monitor")
    logger.info(f"Initial delay: {INITIAL_DELAY} seconds (env: INITIAL_DELAY)")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds (env: CHECK_INTERVAL)") 
    logger.info(f"Inactivity threshold: {INACTIVITY_THRESHOLD} seconds (env: INACTIVITY_THRESHOLD)")
    logger.info(f"Monitoring log file: {LOG_FILE}")
    logger.info(f"Monitor log file: {MONITOR_LOG_FILE}")
    logger.info("="*50)
    
    logger.info(f"Waiting for {INITIAL_DELAY} seconds to allow application boot...")
    time.sleep(INITIAL_DELAY)
    logger.info("Starting log-based inactivity monitoring.")

    inactive_time = 0
    while True:
        if has_recent_activity():
            if inactive_time > 0:  # Only log if we were previously inactive
                logger.info(f"Pod became active again after {inactive_time} seconds of inactivity")
            inactive_time = 0  # Reset if activity detected
            logger.debug("Pod active (recent log activity detected)")
        else:
            inactive_time += CHECK_INTERVAL
            logger.info(f"Inactive for {inactive_time} seconds (no recent log activity)")
            if inactive_time >= INACTIVITY_THRESHOLD:
                logger.critical("Inactivity threshold reached. Stopping pod.")
                stop_pod()
                break
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
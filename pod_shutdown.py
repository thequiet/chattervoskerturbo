# filepath: /Users/jeffcallahan/chattervosker/pod_shutdown.py
import os
import sys
import logging
from datetime import datetime

try:
    import torch
except Exception:  # torch may be unavailable in some environments
    torch = None

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

_shutdown_done = False


def _get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    if logger is not None:
        return logger
    lg = logging.getLogger("pod_shutdown")
    if not lg.handlers:
        logging.basicConfig(level=logging.INFO)
    return lg


def _get_s3_client():
    S3_REGION = os.environ.get("S3_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    AWS_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")

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
    return session.client("s3", **client_kwargs)


def _s3_upload(path: str, logger: logging.Logger, key_prefix: str):
    bucket = os.environ.get("S3_BUCKET") or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        logger.debug("S3 bucket not configured; skipping upload")
        return None

    try:
        client = _get_s3_client()
    except Exception as e:
        logger.error(f"Init S3 client failed: {e}")
        return None

    try:
        basename = os.path.basename(path)
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        key = "/".join([p.strip("/") for p in [key_prefix, date_prefix, basename] if p])
        logger.info(f"Uploading {path} to s3://{bucket}/{key}")
        client.upload_file(path, bucket, key)
        return {"bucket": bucket, "key": key}
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected S3 upload error: {e}")
    return None


def _cleanup_network_volume(logger: logging.Logger):
    """
    Clean up network volume if needed
    """
    try:
        # Import here to avoid dependency issues
        from setup_network_volume import RunPodNetworkVolumeManager
        
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            logger.debug("RUNPOD_API_KEY not set; skipping network volume cleanup")
            return
        
        volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "inmf69hxr7")
        logger.info(f"Attempting to unmount network volume {volume_id}")
        
        manager = RunPodNetworkVolumeManager(api_key)
        success = manager.unmount_network_volume(volume_id)
        
        if success:
            logger.info("✓ Network volume unmounted successfully")
        else:
            logger.warning("Network volume unmount may have failed")
            
    except ImportError:
        logger.debug("Network volume manager not available; skipping cleanup")
    except Exception as e:
        logger.warning(f"Network volume cleanup failed: {e}")


def perform_pod_shutdown(shutdown_reason: str | None = None,
                          logger: logging.Logger | None = None,
                          extra_paths: list[str] | None = None) -> bool:
    """
    Centralized shutdown routine for the pod.
    - Clears CUDA cache if available
    - Uploads logs to S3 if configured
    - Idempotent across multiple invocations
    """
    global _shutdown_done
    lg = _get_logger(logger)

    if _shutdown_done:
        lg.debug("Shutdown already performed; skipping")
        return False
    _shutdown_done = True

    lg.info("Starting pod shutdown routine")
    if shutdown_reason:
        lg.info(f"Reason: {shutdown_reason}")

    # Clear CUDA cache if available
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            lg.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
    except Exception as e:
        lg.warning(f"Failed to clear CUDA cache: {e}")

    # Clean up network volume
    _cleanup_network_volume(lg)

    # Upload logs
    try:
        prefix = (os.environ.get("S3_PREFIX") or os.environ.get("S3_DIR") or "uploads").strip("/")
        key_prefix = f"{prefix}/logs"
        default_paths = [
            "/app/app.log",
            "/app/inactivity_monitor.log",
        ]
        paths = []
        for p in default_paths + (extra_paths or []):
            if p and p not in paths:
                paths.append(p)

        for p in paths:
            try:
                if p and os.path.exists(p):
                    _s3_upload(p, lg, key_prefix=key_prefix)
            except Exception as e:
                lg.error(f"Failed uploading {p}: {e}")
    except Exception as e:
        lg.error(f"Error during log upload step: {e}")

    lg.info("Pod shutdown routine finished")
    return True


if __name__ == "__main__":
    reason = os.environ.get("SHUTDOWN_REASON") or "cli"
    extras = os.environ.get("EXTRA_LOGS")
    extra_list = [x.strip() for x in extras.split(",") if x.strip()] if extras else None
    performed = perform_pod_shutdown(shutdown_reason=reason, extra_paths=extra_list)
    sys.exit(0 if performed else 0)

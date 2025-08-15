#!/usr/bin/env python3
"""
RunPod Network Volume Setup Script
Handles mounting and unmounting of network volumes via RunPod API
"""

import os
import requests
import json
import time
import logging
import sys
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RunPod configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
NETWORK_VOLUME_ID = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "inmf69hxr7")
DATA_CENTER_REGION = os.environ.get("RUNPOD_DATA_CENTER_REGION", "US-IL-1")
MOUNT_PATH = os.environ.get("RUNPOD_MOUNT_PATH", "/network-storage")

# Log configuration values
logger.info("=== RunPod Configuration ===")
logger.info(f"NETWORK_VOLUME_ID: {NETWORK_VOLUME_ID}")
logger.info(f"DATA_CENTER_REGION: {DATA_CENTER_REGION}")
logger.info(f"MOUNT_PATH: {MOUNT_PATH}")
logger.info(f"API_KEY present: {'Yes' if RUNPOD_API_KEY else 'No'}")
logger.info("===============================")

class RunPodNetworkVolumeManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def _make_request(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Make a GraphQL request to RunPod API"""
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            
            logger.debug(f"API Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            logger.debug(f"API Response status: {response.status_code}")
            logger.debug(f"API Response text: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"RunPod API request failed: {e}")
            logger.error(f"Response content: {getattr(e.response, 'text', 'No response content') if hasattr(e, 'response') else 'No response object'}")
            return None
    
    def get_pod_id(self) -> Optional[str]:
        """Get the current pod ID from environment or metadata"""
        # Try environment variable first
        pod_id = os.environ.get("RUNPOD_POD_ID")
        if pod_id:
            return pod_id
        
        # Try to get from RunPod metadata endpoint
        try:
            response = requests.get("http://metadata.runpod.io/pod", timeout=5)
            if response.status_code == 200:
                metadata = response.json()
                return metadata.get("id")
        except Exception as e:
            logger.warning(f"Could not fetch pod metadata: {e}")
        
        return None
    
    def mount_network_volume(self, volume_id: str, mount_path: str) -> bool:
        """Mount a network volume to the current pod"""
        pod_id = self.get_pod_id()
        if not pod_id:
            logger.error("Could not determine pod ID")
            return False
        
        # Try the new GraphQL mutation format first
        query = """
        mutation mountNetworkVolume($input: MountNetworkVolumeInput!) {
            mountNetworkVolume(input: $input) {
                id
                status
                errors {
                    message
                    field
                }
            }
        }
        """
        
        variables = {
            "input": {
                "podId": pod_id,
                "networkVolumeId": volume_id,
                "mountPath": mount_path,
                "dataCenterRegion": DATA_CENTER_REGION
            }
        }
        
        logger.info(f"Mounting network volume {volume_id} to {mount_path} in region {DATA_CENTER_REGION}")
        logger.info(f"Pod ID: {pod_id}")
        result = self._make_request(query, variables)
        
        if result and not result.get("errors"):
            mount_result = result.get("data", {}).get("mountNetworkVolume", {})
            if mount_result.get("errors"):
                logger.error(f"Mount operation returned errors: {mount_result['errors']}")
                return False
            logger.info("Network volume mount request submitted successfully")
            return self._wait_for_mount(mount_path)
        else:
            logger.error(f"Failed to mount network volume: {result}")
            
            # Try alternative mutation format without dataCenterRegion
            logger.info("Trying alternative mutation format...")
            query_alt = """
            mutation mountNetworkVolume($input: MountNetworkVolumeInput!) {
                mountNetworkVolume(input: $input) {
                    id
                    status
                }
            }
            """
            
            variables_alt = {
                "input": {
                    "podId": pod_id,
                    "networkVolumeId": volume_id,
                    "mountPath": mount_path
                }
            }
            
            result_alt = self._make_request(query_alt, variables_alt)
            if result_alt and not result_alt.get("errors"):
                logger.info("Network volume mount request submitted successfully (alternative format)")
                return self._wait_for_mount(mount_path)
            else:
                logger.error(f"Alternative mount request also failed: {result_alt}")
                return False
    
    def _wait_for_mount(self, mount_path: str, timeout: int = 300) -> bool:
        """Wait for network volume to be mounted"""
        logger.info(f"Waiting for {mount_path} to be mounted...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.ismount(mount_path) or os.path.exists(mount_path):
                # Additional check - try to create a test file
                try:
                    os.makedirs(mount_path, exist_ok=True)
                    test_file = os.path.join(mount_path, ".mount_test")
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                    logger.info(f"Network volume successfully mounted at {mount_path}")
                    return True
                except Exception as e:
                    logger.debug(f"Mount test failed: {e}")
            
            time.sleep(5)
        
        logger.error(f"Timeout waiting for network volume to mount at {mount_path}")
        return False
    
    def unmount_network_volume(self, volume_id: str) -> bool:
        """Unmount a network volume from the current pod"""
        pod_id = self.get_pod_id()
        if not pod_id:
            logger.error("Could not determine pod ID")
            return False
        
        query = """
        mutation unmountNetworkVolume($input: UnmountNetworkVolumeInput!) {
            unmountNetworkVolume(input: $input) {
                id
                status
            }
        }
        """
        
        variables = {
            "input": {
                "podId": pod_id,
                "networkVolumeId": volume_id
            }
        }
        
        logger.info(f"Unmounting network volume {volume_id}")
        result = self._make_request(query, variables)
        
        if result and not result.get("errors"):
            logger.info("Network volume unmount request submitted successfully")
            return True
        else:
            logger.error(f"Failed to unmount network volume: {result}")
            return False

def mount_network_volume() -> bool:
    """Mount the network volume using RunPod API"""
    if not RUNPOD_API_KEY:
        logger.error("RUNPOD_API_KEY environment variable not set")
        return False
    
    manager = RunPodNetworkVolumeManager(RUNPOD_API_KEY)
    return manager.mount_network_volume(NETWORK_VOLUME_ID, MOUNT_PATH)

def ensure_network_storage_available() -> bool:
    """Ensure network storage is available, mount if necessary"""
    # Check if already mounted
    if os.path.exists(MOUNT_PATH) and os.path.isdir(MOUNT_PATH):
        try:
            # Test write access
            test_file = os.path.join(MOUNT_PATH, ".access_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Network storage already available at {MOUNT_PATH}")
            return True
        except Exception as e:
            logger.warning(f"Network storage exists but not writable: {e}")
    
    # Try to mount
    logger.info("Attempting to mount network storage...")
    return mount_network_volume()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mount":
        success = ensure_network_storage_available()
        sys.exit(0 if success else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "unmount":
        if RUNPOD_API_KEY:
            manager = RunPodNetworkVolumeManager(RUNPOD_API_KEY)
            success = manager.unmount_network_volume(NETWORK_VOLUME_ID)
            sys.exit(0 if success else 1)
        else:
            logger.error("RUNPOD_API_KEY not set")
            sys.exit(1)
    else:
        print("Usage: python setup_network_volume.py [mount|unmount]")
        sys.exit(1)

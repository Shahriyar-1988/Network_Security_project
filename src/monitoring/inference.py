# Activate only when testing this code individually
import sys
import os

# ✅ Fix path so that 'src' can be imported properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.monitoring.drift_check import DriftCheck
import pandas as pd
import os
from src.logging.logger import logger
import yaml

def run_drift_check():
    # Load new incoming data ( as a show case I will load the test data)
    # --- Step 1: Get the latest timestamped artifact folder ---
    ARTIFACTS_BASE_DIR = "artifacts"

    # List subdirectories and get the latest by modification time
    subdirs = [
        os.path.join(ARTIFACTS_BASE_DIR, d) for d in os.listdir(ARTIFACTS_BASE_DIR)
        if os.path.isdir(os.path.join(ARTIFACTS_BASE_DIR, d))
    ]

    if not subdirs:
        raise FileNotFoundError("No timestamped artifact directories found.")

    latest_artifact_dir = max(subdirs, key=os.path.getmtime)
    logger.info(f"Using latest artifact directory: {latest_artifact_dir}")
    REFERENCE_DATA_PATH = os.path.join(latest_artifact_dir,"data_validation", "validated", "train.csv")
    INCOMING_DATA_PATH = os.path.join(latest_artifact_dir, "data_validation","validated", "test.csv")
    DRIFT_REPORT_PATH = os.path.join(latest_artifact_dir,"data_validation", "drift_report.yaml")
    #  --- Step 3: Drift check ---
    drift_checker = DriftCheck(reference_data_path=REFERENCE_DATA_PATH)
    new_data = pd.read_csv(INCOMING_DATA_PATH)

    drift_report, drift_found = drift_checker.detect_drift(new_data)

    if drift_found:
        logger.info(" Drift Detected!")
    else:
        logger.info(" No Drift Detected.")

    with open(DRIFT_REPORT_PATH, "w") as f:
        yaml.dump(drift_report, f)

    logger.info(f" Drift report saved to: {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    run_drift_check()
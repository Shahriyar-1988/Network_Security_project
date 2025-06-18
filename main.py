"""Main pipeline orchestration script for Network Security project."""
# import os
# import sys

# #  Ensure your project root is on the Python path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.training_pipeline import TrainingPipeline
from src.logging.logger import logger
try:
    logger.info("Starting the Network Security Training Pipeline")
    pipeline=TrainingPipeline()
    pipeline.execute_pipeline()
    logger.info("Training Pipeline executed successfully")
except Exception as e:
    logger.error(f"Training Pipeline failed due to error: {e}")
    raise e


    
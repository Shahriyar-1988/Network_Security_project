"""Main pipeline orchestration script for Network Security project."""
import os
import sys

#  Ensure your project root is on the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from src.entity.config_entity import (TrainingConfig, DataIngestionConfig, 
                                      DataValidationConfig, 
                                      DataTransformationConfig,
                                        ModelTrainerConfig)
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.logging.logger import logger


if __name__ == "__main__":
    try:
        logger.info(">>>> Pipeline Execution Started <<<<")
        
        # 1. Training Configuration (Global Timestamped Setup)
        training_config = TrainingConfig()

        # 2. Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
        logger.info(" Data Ingestion Completed.")

        # 3. Data Validation
        data_validation_config = DataValidationConfig(training_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact: DataValidationArtifact = data_validation.initiate_data_validation()
        logger.info(" Data Validation Completed.")

        # 4. Data Transformation
        data_transformation_config = DataTransformationConfig(training_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact: DataTransformationArtifact = data_transformation.initiate_data_transformation()
        logger.info(" Data Transformation Completed.")

        # 5. Model Training
        model_trainer_config = ModelTrainerConfig(training_config)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact: ModelTrainerArtifact = model_trainer.initiate_model_training()
        logger.info(" Model Training Completed.")


        # 6. Model Evaluation
        model_evaluator = ModelEvaluation(
            data_transformation_artifact=data_transformation_artifact,
            model_evaluation_config=model_trainer_artifact
        )
        model_evaluator.initiate_model_evaluation()
        logger.info(" Model Evaluation & Final Model Saving Completed.")

        logger.info(">>>>  Pipeline Execution Finished Successfully  <<<<")

    except Exception as e:
        logger.error(f" Pipeline execution failed due to error: {e}")
        raise e
    
    
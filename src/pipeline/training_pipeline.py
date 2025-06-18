import os
from src.logging.logger import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import *
from src.entity.artifact_entity import *
 
class TrainingPipeline:
    def __init__(self)->DataIngestionArtifact:
        training_pipeline_config=TrainingConfig()
        self.data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        self.data_validation_config=DataValidationConfig(training_pipeline_config)
        self.data_transformation_config=DataTransformationConfig(training_pipeline_config)
        self.model_trainer_config=ModelTrainerConfig(training_pipeline_config)
    def data_ingestion(self):
        try:
            logger.info(" Data Ingestion Started.")
            data_ingestion_obj=DataIngestion(self.data_ingestion_config)
            data_ingeston_artifact=data_ingestion_obj.initiate_data_ingestion()
            logger.info(" Data Ingestion Completed.")
            return data_ingeston_artifact

        except Exception as e:
            raise e
    def data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
         logger.info(" Data Validation Started.")
         data_validation_obj=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                        data_validation_config=self.data_validation_config)
         data_validation_artifact=data_validation_obj.initiate_data_validation()
         logger.info(" Data Validation Completed.")
         return data_validation_artifact
        except Exception:
            raise Exception
    def data_transformation(self, data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            logger.info(" Data Transformation Started.")
            data_transformation_obj=DataTransformation(data_transformation_config=self.data_transformation_config,
                                                    data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=data_transformation_obj.initiate_data_transformation()
            logger.info(" Data Transformation Completed.")
            return data_transformation_artifact
        except Exception:
            raise Exception
        
    def model_trainer(self, data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logger.info(" Model Training Started.")
            model_trainer_obj=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                        model_training_config=self.model_trainer_config)
            model_trainer_artifact=model_trainer_obj.initiate_model_training()
            logger.info(" Model Training Completed.")
            return model_trainer_artifact
        except Exception:
            raise Exception
    def model_evaluation(self, data_transformation_artifact:DataTransformationArtifact,
                         model_trainer_artifact:ModelTrainerArtifact)->None:
        try:
            model_evaluation_obj=ModelEvaluation(data_transformation_artifact=data_transformation_artifact,
                                                model_evaluation_config=model_trainer_artifact)
            model_evaluation_obj.initiate_model_evaluation()
            logger.info(" Model Evaluation & Final Model Saving Completed.")
        except Exception:
            raise Exception
    def execute_pipeline(self):
        try:
            data_ingestion_artifact=self.data_ingestion()
            data_validation_artifact=self.data_validation(data_ingestion_artifact)
            data_transformation_artifact=self.data_transformation(data_validation_artifact) 
            model_trainer_artifact=self.model_trainer(data_transformation_artifact)
            self.model_evaluation(data_transformation_artifact,
                              model_trainer_artifact)
        except Exception:
            raise Exception
    
      

        

        




            
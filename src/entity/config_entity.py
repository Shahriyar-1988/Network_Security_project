# Activate only when testing this code individually
# import sys
# import os

# # ✅ Fix path so that 'src' can be imported properly
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime
import os
from src.constants import (data_ingestion_const,training_const,
                           data_validation_const,
                           data_transformation_const,
                           model_trainer_const)

class TrainingConfig:
    def __init__(self,timestamp=None):
        timestamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.time_stamp:str=timestamp
        self.pipeline_name=training_const.PIPELINE_NAME
        self.artifact_name=training_const.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        

class DataIngestionConfig:
    def __init__(self, config:TrainingConfig):
        self.data_ingestion_dir=os.path.join(
            config.artifact_dir,
            data_ingestion_const.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path=os.path.join(
            self.data_ingestion_dir,
            data_ingestion_const.DATA_INGESTION_FEATURE_STORE_DIR,
            training_const.FILE_NAME
        )
        self.training_file_path = os.path.join(
                self.data_ingestion_dir, 
                data_ingestion_const.DATA_INGESTION_INGESTED_DIR,
                training_const.TRAIN_FILE_NAME)
        self.test_file_path=os.path.join(
                self.data_ingestion_dir, 
                data_ingestion_const.DATA_INGESTION_INGESTED_DIR,
                training_const.TEST_FILE_NAME)
        self.train_test_split_ratio:float=data_ingestion_const.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name=data_ingestion_const.DATA_INGESTION_COLLECTION_NAME
        self.database_name=data_ingestion_const.DATA_INGESTION_DATABASE_NAME

class DataValidationConfig:
    def __init__(self, config:TrainingConfig):
        self.data_validation_dir=os.path.join(config.artifact_dir,data_validation_const.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir=os.path.join(self.data_validation_dir, data_validation_const.DATA_VALIDATION_VALID_DIR)
        self.valid_train_file_path=os.path.join(self.valid_data_dir,training_const.TRAIN_FILE_NAME)
        self.valid_test_file_path=os.path.join(self.valid_data_dir,training_const.TEST_FILE_NAME)
        self.invalid_data_dir=os.path.join(self.data_validation_dir,data_validation_const.DATA_VALIDATION_INVALID_DIR)
        self.invalid_train_file_path=os.path.join(self.invalid_data_dir,training_const.TRAIN_FILE_NAME)
        self.invalid_test_file_path=os.path.join(self.valid_data_dir,training_const.TEST_FILE_NAME)
        self.drift_report_file_path=os.path.join(self.data_validation_dir,
                                                 data_validation_const.DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                 data_validation_const.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
class DataTransformationConfig:
    def __init__(self,config:TrainingConfig):
        self.data_transformation_dir=os.path.join(config.artifact_dir,data_transformation_const.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_data_dir=os.path.join(self.data_transformation_dir,data_transformation_const.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        self.transformed_train_file_path=os.path.join(self.transformed_data_dir,
                                                      data_transformation_const.DATA_TRANSFORMATION_TRAIN_FILE_PATH)
        
        self.transformed_test_file_path = os.path.join(self.transformed_data_dir,
                                                      data_transformation_const.DATA_TRANSFORMATION_TEST_FILE_PATH)
        self.imputation_parameters =data_transformation_const.DATA_TRANSFORMATION_IMPUTER_PARAMS
        self.preprocessing_file_path=os.path.join(self.data_transformation_dir,data_transformation_const.PREPROCESSING_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self,config:TrainingConfig):
        self.model_trainer_dir=os.path.join(config.artifact_dir,
                                            model_trainer_const.MODEL_TRAINER_DIR_NAME)
        self.trained_model_dir=os.path.join(self.model_trainer_dir,
                                            model_trainer_const.MODEL_TRAINER_TRAINED_MODEL_DIR,
                                            model_trainer_const.MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.trainer_expected_accuracy=model_trainer_const.MODEL_TRAINER_MIN_EXPECTED_SCORE
        self.trainer_threshold=model_trainer_const.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD

if __name__ == "__main__":
    training_config = TrainingConfig()
    data_ingestion_config = DataIngestionConfig(training_config)

    print("Artifact Directory:", training_config.artifact_dir)
    print("Data Ingestion Directory:", data_ingestion_config.data_ingestion_dir)
    print("Feature Store Path:", data_ingestion_config.feature_store_file_path)
    print("Training File Path:", data_ingestion_config.training_file_path)
    print("Test File Path:", data_ingestion_config.test_file_path)
    print("Database Name:", data_ingestion_config.database_name)
    print("Collection Name:", data_ingestion_config.collection_name)
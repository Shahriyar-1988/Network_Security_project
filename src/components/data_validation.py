# Activate only when testing this code individually
import sys
import os

# ✅ Fix path so that 'src' can be imported properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.entity.config_entity import DataValidationConfig,TrainingConfig,DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.utils.common import read_yaml, write_yaml_file
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.schema_config = read_yaml(SCHEMA_FILE_PATH)

    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def validate_num_of_columns(self, dataframe: pd.DataFrame) -> bool:
        expected_num_cols = len(self.schema_config["COLUMNS"])
        return expected_num_cols == len(dataframe.columns)

    def validate_numerical_column_exists(self, dataframe: pd.DataFrame) -> bool:
        return len(dataframe.select_dtypes(include=np.number).columns) > 0

    def validate_column_names(self, dataframe: pd.DataFrame) -> bool:
        expected_cols = set(col for item in self.schema_config.COLUMNS for col in item.keys())
        actual_cols = set(dataframe.columns)
        return expected_cols == actual_cols

    def initiate_data_validation(self) -> DataValidationArtifact:
        df_train = self.read_file(self.data_ingestion_artifact.train_file_path)
        df_test = self.read_file(self.data_ingestion_artifact.test_file_path)

        for name, df in [("train", df_train), ("test", df_test)]:
            if not self.validate_num_of_columns(df):
                raise ValueError(f"{name} file does not match expected number of columns.")
            if not self.validate_column_names(df):
                raise ValueError(f"{name} file column names do not match schema.")
            if not self.validate_numerical_column_exists(df):
                raise ValueError(f"No numerical columns found in {name} data.")

        # Save valid files
        os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
        df_train.to_csv(self.data_validation_config.valid_train_file_path, index=False)
        df_test.to_csv(self.data_validation_config.valid_test_file_path, index=False)

        # Create and return artifact
        return DataValidationArtifact(
            validation_status=True,
            valid_train_file_path=self.data_validation_config.valid_train_file_path,
            valid_test_file_path=self.data_validation_config.valid_test_file_path,
            invalid_train_file_path=None,
            invalid_test_file_path=None,
            drift_report_file_path=None  # no drift check at this stage
        )

    
if __name__ == "__main__":
    # Assume you already have a data ingestion artifact (from previous step)
    training_config = TrainingConfig()
    data_validation_config = DataValidationConfig(training_config)

    # artifact paths (you should use real ones after ingestion)
    
    data_ingestion_config = DataIngestionConfig(training_config)
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_path_object=data_ingestion.initiate_data_ingestion()
    train_path=data_path_object.train_file_path
    test_path=data_path_object.test_file_path
    # Set up data ingestion
    data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_path,
                                                    test_file_path=test_path)

    # Instantiate and run the validation pipeline
    validator = DataValidation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_config=data_validation_config
    )

    try:
        validation_artifact = validator.initiate_data_validation()
        print("✅ Data validation completed.")
        print("Validation Status:", validation_artifact.validation_status)
        print("Drift Report Path:", validation_artifact.drift_report_file_path)
        print("Valid Train Path:", validation_artifact.valid_train_file_path)
        print("Valid Test Path:", validation_artifact.valid_test_file_path)
    except Exception as e:
        print("❌ Data validation failed:", str(e))              
      


          

        
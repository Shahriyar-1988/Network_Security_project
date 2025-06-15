import numpy as np
import pandas as pd
from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.constants import SCHEMA_FILE_PATH
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from utils.common import save_bin,read_yaml
from src.logging.logger import logger
class DataTansformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        self.data_validation_artifact= data_validation_artifact
        self.data_transformation_config=data_transformation_config
        self.schema=read_yaml(SCHEMA_FILE_PATH)
    @staticmethod
    def read_file(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)
    def get_data_transformer(self)->Pipeline:
        logger.info( "Entered get_data_trnasformer method of Trnasformation class")
        imputation_params=self.data_transformation_config.imputation_parameters
        transformer:Pipeline = Pipeline([("imputer",
                                 KNNImputer(*imputation_params)),
                                 ])
        logger.info(
                f"Initialise KNNImputer with {imputation_params}"
            )
        return transformer

    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logger.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logger.info("Starting data transformation")
            train_file_path=self.data_validation_artifact.valid_train_file_path
            test_file_path=self.data_validation_artifact.valid_test_file_path
            df_train=DataTansformation.read_file(train_file_path)
            df_test=DataTansformation.read_file(test_file_path)

            target_col=self.schema.TARGET_COLUMN
            X_train=df_train.drop(target_col,axis=1)
            y_train=df_train.target_col
            y_train = y_train.replace(-1,0) # For Binary classification purposes
            X_test=df_test.drop(target_col,axis=1)
            y_test=df_test.target_col
            y_test=y_test.replace(-1,0)
            preprocessor=self.get_data_transformer()
            X_train_transformed=preprocessor.fit_transform(X_train)
            X_test_transformed=preprocessor.transform(X_test)
            train_arr = np.c_[X_train_transformed,
                                np.array(y_train)]
            test_arr = np.c_[X_test_transformed,
                             np.array(y_test)]
            
            save_bin(self.data_transformation_config.transformed_train_file_path,
                     train_arr)
            save_bin(self.data_transformation_config.transformed_test_file_path,
                     test_arr)
            save_bin(self.data_transformation_config.preprocessing_file_path,
                     preprocessor)
            data_transformation_artifact=DataTransformationArtifact(
                transformed_train_data_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_data_path=self.data_transformation_config.transformed_test_file_path,
                transformation_object_path=self.data_transformation_config.preprocessing_file_path
            )
            return data_transformation_artifact


        except Exception as e:
            raise e



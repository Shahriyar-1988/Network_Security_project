# Activate only when testing this code individually
# import sys
# import os

# # ✅ Fix path so that 'src' can be imported properly
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.logging.logger import logger
from src.entity.config_entity import DataIngestionConfig,TrainingConfig
from src.entity.artifact_entity import DataIngestionArtifact
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.constants import SCHEMA_FILE_PATH
from src.utils.common import read_yaml

load_dotenv()
MONGO_URL=os.getenv("MONGO_DB_URL")
class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    def import_collection_as_dataframe(self):
        database_name=self.config.database_name
        collection_name=self.config.collection_name
        self.mongo_client=MongoClient(MONGO_URL)
        collection=self.mongo_client[database_name][collection_name]
        df=pd.DataFrame(list(collection.find()))
        if "_id" in df.columns.to_list():
            df=df.drop(columns=["_id"],axis=1)
        df.replace({"na":np.nan}, inplace=True)
        return df
    def export_data_to_feature_store(self, df:pd.DataFrame):
        feature_store_path=Path(self.config.feature_store_file_path).parent
        feature_store_path.mkdir(parents=True,exist_ok=True)
        df.to_csv(self.config.feature_store_file_path,index=False)
        return df
    def data_splitter(self,df:pd.DataFrame):
        target_col = read_yaml(SCHEMA_FILE_PATH)["TARGET_COLUMN"]

        if  target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        train_set,test_set = train_test_split(df,
                                              test_size=self.config.train_test_split_ratio,
                                              stratify=df[target_col])
        logger.info("Performed train test split on the dataframe")

        logger.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
        dir_path=Path(self.config.training_file_path).parent
        dir_path.mkdir(parents=True,exist_ok=True)
        logger.info(f"Exporting train and test file path.")
            
        train_set.to_csv(
                self.config.training_file_path, index=False, header=True
            )
        test_set.to_csv(
                self.config.test_file_path, index=False, header=True
            )
        logger.info(f"Exported train and test file path.")
        
    def initiate_data_ingestion(self):
        data=self.import_collection_as_dataframe()
        df=self.export_data_to_feature_store(data)
        self.data_splitter(df)
        data_ingestion_artifact=DataIngestionArtifact(train_file_path= self.config.training_file_path,
                                                      test_file_path=self.config.test_file_path)
        return data_ingestion_artifact

if __name__=="__main__":
        training_config=TrainingConfig()
        data_ingestion_config=DataIngestionConfig(training_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


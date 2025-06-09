
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import json
import pandas as pd
import numpy as np
from src.logging.logger import logging

load_dotenv()
uri = os.getenv("MONGO_DB_URL")
class NetworkDataExtract:
    @staticmethod
    def cv_to_json(file_path):
        try:
            data=pd.read_csv(file_path,index_col=None)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
           
        except Exception as e:
            raise e
    @staticmethod
    def insert_data_mongodb(records,database,collection):
        try:
            mongo_client=MongoClient(uri)
            data_base=mongo_client[database]
            collect=data_base[collection]
            result=collect.insert_many(records)
            logging.info(f"Inserted {len(result.inserted_ids)} records into {database}.{collection}")
            return len(result.inserted_ids)
        except Exception as e:
            raise e

if __name__=="__main__":
   FILEPATH="Network_Data\phisingData.csv" 
   DATABASE="SHAH" # Name of the database
   COLLECTION="NetworkData" # Name of the dataset collection
   records=NetworkDataExtract.cv_to_json(FILEPATH)
   no_rec=NetworkDataExtract.insert_data_mongodb(records,DATABASE,COLLECTION)
   print(no_rec)


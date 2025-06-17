import mlflow.sklearn
import numpy as np
import os
from src.utils.common import load_array,load_bin,save_bin
from src.constants import SCHEMA_FILE_PATH
from src.constants.training_const import FINAL_MODEL_PATH
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.utils.ml_utils.classfication_metrics import get_classification_metrics
from src.utils.ml_utils.final_model import FinalModel
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import asdict


class ModelEvaluation:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 model_evaluation_config: ModelTrainerArtifact):
        self.data_transformation_artifact=data_transformation_artifact
        self.model_evaluation_config= model_evaluation_config
    def MLflow_track(self,model,train_classification_report:dict,test_classification_report:dict)->None:
        load_dotenv()
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        if not MLFLOW_TRACKING_URI:
             raise EnvironmentError("MLFLOW_TRACKING_URI is not set in the environment variables.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_name=self.model_evaluation_config.trained_model_name
        mlflow.set_experiment(f"{model_name} Network Security experiment")
        with mlflow.start_run(run_name=type(model).__name__):
            # Log model parameters
            mlflow.log_param("model_type", type(model).__name__)
            if hasattr(model, 'get_params'):
                for param, value in model.get_params().items():
                     mlflow.log_param(param, value)
        # Log training metrics
        for metric_name, value in train_classification_report.items():
            mlflow.log_metric(f"train_{metric_name}", value)
       

        # Log test metrics
        for metric_name, value in test_classification_report.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        mlflow.sklearn.log_model(model_name,
                                 artifact_path=self.model_evaluation_config.trained_model_file_path)
    @staticmethod
    def save_final_model(preprocessor_path:str, 
                         model: object,
                         save_path:str):
        preprocessor=load_bin(preprocessor_path)
        final_model=FinalModel(preprocessor=preprocessor,
                               model=model)
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        save_bin(save_path, final_model)
        return final_model        
        
    def initiate_model_evaluation(self):
        test_data_path=self.data_transformation_artifact.transformed_test_data_path
        test_data=load_array(test_data_path)
        X_test =test_data[:,:-1]
        y_test=test_data[:,-1]
        trained_model_path=self.model_evaluation_config.trained_model_file_path
        model=load_bin(trained_model_path)
        y_pred=model.predict(X_test)
        test_classification_report = get_classification_metrics(y_test,y_pred)
        self.MLflow_track(model=model,
                          train_classification_report=asdict(self.model_evaluation_config.trained_model_metric_artifact),
                          test_classification_report=asdict(test_classification_report)
        )
        # Saving the finally tested model for deployment into artifacts
        ModelEvaluation.save_final_model(preprocessor_path=self.data_transformation_artifact.transformation_object_path,
                                         model=model,
                                         save_path=FINAL_MODEL_PATH)
        
    

        








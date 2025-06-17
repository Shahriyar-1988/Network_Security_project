from src.logging.logger import logger
import os
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import *
from src.utils.ml_utils.classfication_metrics import get_classification_metrics
from src.utils.ml_utils.model_fitter import get_fitting_report
from src.constants import PARAMS_FILE_PATH
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              AdaBoostClassifier,
                              GradientBoostingClassifier)

class ModelTrainer:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 model_training_config: ModelTrainerConfig):
        self.data_trans_artifact=data_transformation_artifact
        self.model_train_config=model_training_config
        self.params=read_yaml(PARAMS_FILE_PATH)

    def train_model(self,X_train,y_train):
        models = {
                "Random Forest": RandomForestClassifier(random_state=0,verbose=1),
                "Decision Tree": DecisionTreeClassifier(random_state=0),
                "Gradient Boosting": GradientBoostingClassifier(random_state=0,verbose=1),
                "AdaBoost": AdaBoostClassifier(random_state=0),
                "Logistic Regression": LogisticRegression(verbose=1),
            }
        params=self.params
        model_report:dict=get_fitting_report(X_train,y_train,models,params)
        non_overfit_models = {
        name: details
        for name, details in model_report.items()
        if not details["overfitting"]
    }

    # Decide best model based on f1_score
        if non_overfit_models:
            best_model_name = max(non_overfit_models, key=lambda x: non_overfit_models[x]['f1_score'])
            logger.info(f" Selected non-overfitting model: {best_model_name}")
        else:
        # Fall back to overfitting model with best score
            logger.warning(" All models are overfitting. Proceeding with the max score only.")
            best_model_name = max(model_report, key=lambda x: model_report[x]['f1_score'])
        best_score = model_report[best_model_name]['f1_score'] 
        logger.info(f"The best training score on this dataset is {best_score}")
        expected_score=self.model_train_config.trainer_expected_accuracy
        if best_score<expected_score:
            logger.info(f"None of the models could meet the expected accuracy of {expected_score}. Best score: {best_score}")
            raise ValueError("Model training stopped: No model met the expected performance threshold.")
        best_model = model_report[best_model_name]['best_model']
        y_pred=best_model.predict(X_train)
        metrics_report=get_classification_metrics(y_train,y_pred)
        return best_model,best_model_name,metrics_report

    def initiate_model_training(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_trans_artifact.transformed_train_data_path
            train_data=load_array(train_file_path)
            X_train =train_data[:,:-1]
            y_train=train_data[:,-1]
            best_model,best_model_name,metrics_report = self.train_model(X_train,y_train)
            trained_model_path= self.model_train_config.trained_model_dir
            os.makedirs(os.path.dirname(trained_model_path),exist_ok=True)
            save_bin(trained_model_path,best_model)
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_metric_artifact=metrics_report,
                trained_model_file_path=trained_model_path,
                trained_model_name=best_model_name
            )
            return model_trainer_artifact
     
        except Exception as e:
            raise e
        
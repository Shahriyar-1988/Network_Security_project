from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str  
@dataclass
class DataTransformationArtifact:
    transformed_train_data_path: str
    transformed_test_data_path: str
    transformation_object_path: str
@dataclass
class ClassificationArtifact:
    accuracy: float
    f1_score:float
    recall_score:float
    precisioon_score:float
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    trained_model_metric_artifact: ClassificationArtifact
    
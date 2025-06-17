from src.entity.artifact_entity import ClassificationArtifact
from src.logging.logger import logger
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from ensure import ensure_annotations
import numpy as np
@ensure_annotations
def get_classification_metrics(y_true:np.ndarray,y_pred:np.ndarray)->ClassificationArtifact:
    f1=f1_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred)
    accuracy=accuracy_score(y_true,y_pred)
    return ClassificationArtifact(precisioon_score=precision,
                                  accuracy=accuracy,
                                  f1_score=f1,
                                  recall_score=recall 
    )
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from src.logging.logger import logger
from box import ConfigBox
from src.constants.model_trainer_const import MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
from ensure import ensure_annotations
from sklearn.model_selection import train_test_split

@ensure_annotations
def get_fitting_report(X_train,y_train,models:dict,params:ConfigBox)->dict:
    logger.info("Started model training using model_fitter function.")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    report = {}
    for model_name,model in models.items():
        logger.info(f"Training started for model: {model_name}")
        param_grid=params.get(model_name,{})
        if isinstance(param_grid,list):
            param_grid={}
        gs=GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring="f1_weighted",cv=5,
                        n_jobs=-1)
        gs.fit(X_train_split,y_train_split)
        tuned_model=gs.best_estimator_
        best_params=gs.best_params_
        y_train_pred = tuned_model.predict(X_train_split)
        y_val_pred=tuned_model.predict(X_val)
        train_f1 = f1_score(y_train_split, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')

        f1_gap = abs(train_f1 - val_f1)
        overfitting_flag = f1_gap > MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD

        logger.info(f"{model_name} - Train F1: {train_f1:.3f}, Val F1: {val_f1:.3f}, F1 Gap: {f1_gap:.3f}")
        report[model_name]={
            "f1_score": val_f1,
            "train_f1": train_f1,
            "f1_gap": f1_gap,
            "overfitting": overfitting_flag,
            "best_params": best_params,
            "best_model": tuned_model
        }
    logger.info("Completed model training for all models.")
    return report

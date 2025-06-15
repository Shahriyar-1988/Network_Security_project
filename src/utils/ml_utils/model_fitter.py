from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from src.logging.logger import logger
from ensure import ensure_annotations

@ensure_annotations
def get_fitting_report(X_train,y_train,models:dict,params:dict)->dict:
    logger.info("Started model training using model_fitter function.")
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
        gs.fit(X_train,y_train)
        tuned_model=gs.best_estimator_
        best_params=gs.best_params_
        y_train_pred = tuned_model.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        logger.info(f"{model_name} - Training F1 Score: {train_f1:.3f}")
        report[model_name]={
            "f1_score":train_f1,
            "best_params": best_params
        }
    logger.info("Completed model training for all models.")
    return report

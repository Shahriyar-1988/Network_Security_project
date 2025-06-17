from dataclasses import dataclass
@dataclass
class FinalModel:
    preprocessor: object
    model: object
    def predict(self,X):
        X_processed=self.preprocessor.transform(X)
        return self.model.predict(X_processed)

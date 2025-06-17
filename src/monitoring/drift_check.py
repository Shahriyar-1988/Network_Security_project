import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

class DriftCheck:
    def __init__(self, reference_data_path: str, threshold: float = 0.05):
        self.reference_data_path = reference_data_path
        self.threshold = threshold
        self.reference_df = pd.read_csv(reference_data_path)

    def detect_drift(self, new_df: pd.DataFrame) -> tuple[dict, bool]:
        drift_report = {}
        drift_found = False

        # Select only numeric columns from reference data
        numerical_cols = self.reference_df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            if col in new_df.columns:
                ref_data = self.reference_df[col].dropna()
                new_data = new_df[col].dropna()

                # Perform KS test
                _, p_value = ks_2samp(ref_data, new_data)
                drifted = p_value < self.threshold

                # Use built-in types only
                drift_report[col] = {
                    "p_value": float(p_value),
                    "drifted": bool(drifted)
                }

                if drifted:
                    drift_found = True
            else:
                drift_report[col] = {"error": "Column missing in new data"}

        return drift_report, drift_found

# Activate only when testing this code individually
import sys
import os

# ✅ Fix path so that 'src' can be imported properly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os
import glob
from src.utils.common import load_bin
import pandas as pd
from datetime import datetime
from src.logging.logger import logger
from src.constants.training_const import FINAL_MODEL_PATH

def run_batch_prediction(input_dir: str, output_dir: str = "batch_predictions") -> str:
    logger.info(f"Starting batch prediction from {input_dir}")
    
    trained_model = load_bin(FINAL_MODEL_PATH)
    logger.info("FinalModel loaded successfully")

    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in input directory.")
        return None  #  important to explicitly return None

    # For simplicity, handle only the first CSV file (for Streamlit)
    input_path = csv_files[0]
    df = pd.read_csv(input_path)
    df.drop(columns=["Result"],axis=1,inplace=True)
    logger.info(f"Predicting on file: {os.path.basename(input_path)}")

    try:
        preds = trained_model.predict(df)
        df["prediction"] = preds
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise e

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_{timestamp}.csv"
    out_path = os.path.join(output_dir, out_name)

    df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to: {out_path}")
    logger.info("Batch prediction completed!")
    
    return out_path  #  Proper return after processing

if __name__=="__main__":
    run_batch_prediction(input_dir="Artifacts\\06_18_2025_12_37_31\data_validation\\validated\\")

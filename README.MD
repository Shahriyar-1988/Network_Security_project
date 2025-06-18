
# 🔐 Network Security – End-to-End ML Pipeline

This is a showcase project to present my abilities in the development and deployment of an end-to-end machine learning pipeline, with a focus on **cybersecurity and malicious URL detection**. The project demonstrates my skills across the ML lifecycle — from data ingestion to production-ready deployment.

---

## 🚀 Project Highlights

- **Modular Pipeline** using custom components for:
  - ✅ Data ingestion, validation, and transformation  
  - ✅ Model training, tuning (via `GridSearchCV`), and evaluation  
  - ✅ Overfitting/underfitting checks and drift detection  
  - ✅ MLflow logging for experiment tracking  
  - ✅ Batch prediction support for incoming CSVs  
  - ✅ Streamlit app for interactive use

- **Preprocessing**
  - Missing value imputation with `KNNImputer`
  - Feature scaling & label normalization
  - YAML schema-driven pipeline logic

- **Modeling**
  - Ensemble methods (`RandomForest`, `GradientBoosting`, `AdaBoost`) and `LogisticRegression`
  - Custom evaluation metrics with `f1_score`, precision, recall, accuracy

- **Monitoring**
  - Data drift detection using `Kolmogorov–Smirnov` test
  - Drift reports saved in timestamped YAML files

- **Deployment**
  - ✅ Final model serialization (including preprocessor)  
  - ✅ `batch_prediction.py` for real-world inference  
  - ✅ Streamlit app for CSV-based prediction and visualization

---

## 📁 Folder Structure

```
.
├── src/
│   ├── components/         # Data & model pipeline steps
│   ├── utils/              # Utility functions
│   ├── entity/             # Config & artifact classes
│   ├── constants/          # Static paths and values
│   ├── pipeline/           # Training & prediction pipelines
│   └── monitoring/         # Drift checking logic
├── artifacts/              # Timestamped pipeline outputs
├── app.py                  # Streamlit app
├── batch_prediction.py     # Inference logic
├── main.py                 # Training pipeline trigger
└── requirements.txt
```

---

## 🧪 Run Locally

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Trigger training pipeline:**
```bash
python main.py
```

**Run Streamlit app for prediction:**
```bash
streamlit run app.py
```

---

## 📊 MLflow Tracking

Track all training metrics, parameters, and models via MLflow by setting:
```env
MLFLOW_TRACKING_URI=<your_tracking_uri>
```

---

## 🎯 Purpose

Built to simulate a production-grade ML system in the security domain. This project reflects real-world challenges like data quality, model drift, and deployment readiness — all handled with modular, testable, and extensible code.

---

Feel free to explore, fork, or ask questions!  
> **Author**: Shahriyar A. | 2025

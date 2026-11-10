# Predictive Maintenance for Industrial Machines

Machine learning project that predicts industrial machine failure using operational and sensor data, built as part of the CoderAxo AI/ML internship.

## Overview

Unplanned machine failures cause costly downtime. This project trains and compares classification models to predict whether a machine will fail, using the **AI4I 2020 Predictive Maintenance Dataset** (10,000 records, 14 attributes). The trained model is deployed as a **FastAPI web application** with a full frontend, SQLite prediction history, and CSV/PDF export.

## Dataset

- **Source:** AI4I 2020 Predictive Maintenance Dataset
- **File:** `ai4i2020.csv` — place it inside the `Dataset/` folder
- **Target:** `Machine failure` (binary: 0 = no failure, 1 = failure)

## Project Structure

```
Project_Name/
├── Notebook.ipynb
├── Report.pdf
├── README.md
├── requirements.txt
├── Dataset/
│   └── ai4i2020.csv
├── Output_Images/
├── Demo_Video.mp4
└── Source_Code/
    ├── main.py
    ├── run_once_retrain.py
    ├── models/               ← auto-created after running run_once_retrain.py
    ├── templates/
    │   ├── index.html
    │   └── history.html
    └── static/
        └── css/
            └── style.css
```

## Project Workflow

1. Data loading and exploratory data analysis (EDA)
2. Data cleaning (dropping identifiers and failure-mode flags to avoid data leakage)
3. Encoding categorical features and feature scaling
4. Feature engineering (temperature difference feature)
5. Model training: Logistic Regression, Random Forest, Gradient Boosting
6. Hyperparameter tuning (GridSearchCV on the best model)
7. Model evaluation: accuracy, precision, recall, F1-score, confusion matrix
8. Model deployment via FastAPI with prediction history and export features

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 0.973 | 0.372 |
| Random Forest | 0.987 | 0.750 |
| Gradient Boosting | 0.986 | 0.759 |
| **Optimized Random Forest** | **0.9875** | **0.762** |

The tuned **Random Forest Classifier** was selected for deployment.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
fastapi
uvicorn
scikit-learn
pandas
numpy
jinja2
python-multipart
reportlab
aiofiles
```

## Usage

### 1. Train and save the model

Navigate into `Source_Code/` and run the retrain script once:

```bash
cd Source_Code
python run_once_retrain.py
```

This reads `../Dataset/ai4i2020.csv`, trains the optimized Random Forest, and saves the model artifacts into `Source_Code/models/`.

### 2. Start the web application

```bash
uvicorn main:app --reload
```

Open your browser at **http://127.0.0.1:8000**

### 3. Use the app

| Page | URL | Description |
|---|---|---|
| Predict | `/` | Enter sensor readings and get a real-time prediction |
| History | `/history` | View all past predictions stored in SQLite |
| Export CSV | `/export/csv` | Download full prediction history as CSV |
| Export PDF | `/export/pdf` | Download full prediction history as PDF |

### Input Fields

| Field | Range |
|---|---|
| Machine Type | H / M / L |
| Air Temperature | 290 – 310 K |
| Process Temperature | 295 – 315 K (must be > Air Temp) |
| Rotational Speed | 1000 – 3000 rpm |
| Torque | 0 – 100 Nm |
| Tool Wear | 0 – 300 min |

## Application Features

- Real-time machine failure prediction with failure probability %
- Input validation and descriptive error messages
- Color-coded result (🟢 No Failure / 🔴 Failure)
- SQLite database storing all prediction records
- Export prediction history as CSV or PDF
- Responsive professional UI

## Tech Stack

Python · FastAPI · Jinja2 · SQLite · pandas · NumPy · scikit-learn · ReportLab · HTML/CSS

## Future Work

- Real-time sensor data integration for continuous monitoring
- Try XGBoost / LightGBM / deep learning models
- Remaining Useful Life (RUL) prediction
- Explainable AI (SHAP / LIME) for feature importance visualization

## Author

**Abdullah Shafiq**
Intern ID: CAX-OL-2026-275 — CoderAxo AI/ML Internship

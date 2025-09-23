import pandas as pd
from model.save_load import load_model
from pipeline.preprocessing import apply_tfidf
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

MODEL_PATH = os.path.join(BASE_DIR, "test", "rf_calibrated_model.joblib")
PIPELINE_PATH = os.path.join(BASE_DIR, "test", "tfidf_pipeline.joblib")
COLUMNS_PATH = os.path.join(BASE_DIR, "test", "train_columns.joblib")

rf_calibrated = load_model(MODEL_PATH)
tfidf_pipeline = load_model(PIPELINE_PATH)
train_columns = load_model(COLUMNS_PATH)

def predict_new_data(df: pd.DataFrame, threshold: float = 0.5):
    """Preprocess new data and predict"""
    df_final, _ = apply_tfidf(
        df,
        column="event_info",
        tfidf=tfidf_pipeline,
        train_columns=train_columns
    )

    # drop helper column
    X = df_final.drop(columns=['clean_text'], errors='ignore')

    # 🔹 Ensure we have exactly the training schema (avoid duplicates / missing)
    X = X.reindex(columns=train_columns, fill_value=0)

    # predict
    y_proba = rf_calibrated.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # attach predictions
    df_final['prediction'] = y_pred
    df_final['probability'] = y_proba

    return df_final
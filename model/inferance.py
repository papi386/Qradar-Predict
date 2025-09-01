import pandas as pd
from model.save_load import load_model
from pipeline.preprocessing import apply_tfidf


rf_calibrated = load_model("rf_calibrated.joblib")
tfidf_pipeline = load_model("tfidf_pipeline.joblib")
train_columns = load_model("train_columns.joblib")


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

import pandas as pd
from pipeline.preprocessing import apply_tfidf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "training_data.csv")

train = pd.read_csv(csv_path)

# --- Apply pipeline on training data first ---
train_final, tfidf_vectorizer = apply_tfidf(
    train, column="event_info"
)

# ✅ Save only the *transformed* feature names
TRAIN_COLUMNS = train_final.columns.tolist()

def apply_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing + TF-IDF pipeline (for new/test data)"""
    df_final, _ = apply_tfidf(
        df,
        column="event_info",
        tfidf=tfidf_vectorizer,      # reuse same vectorizer
        train_columns=TRAIN_COLUMNS  # align only to transformed columns
    )
    df_final=df_final.drop(columns='Unnamed: 0' )
    return df_final

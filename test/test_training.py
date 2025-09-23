
import pandas as pd
from model.train import train_model, evaluate_model
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path_train = os.path.join(BASE_DIR, "data", "training_data.csv")


if __name__ == "__main__":
    # Load your train/test datasets
    df_train = pd.read_csv(csv_path_train)
    df_train=df_train.drop(columns=['Unnamed: 0'])

    # Train model
    model, tfidf_pipeline, train_columns = train_model(df_train)
    print("✅ Training completed.")

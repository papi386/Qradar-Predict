import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, classification_report

from model.save_load import save_model
from pipeline.preprocessing import apply_tfidf


def train_model(df, target_column="criticite"):
    """Train RF with TF-IDF, calibrate, save, and return model and tfidf pipeline"""
    # === Step 1: Apply TF-IDF to training data ===
    X = df.drop(columns=target_column)
    print(X.head())
    y = df[target_column].astype(int)  # single column, numeric
    print(y)
    X_final, tfidf_pipeline = apply_tfidf(X, column="event_info")
    train_columns = X_final.columns.tolist()

    # === Step 2: Train RandomForest ===
    best_params = {'max_depth': 33, 'min_samples_leaf': 1,
                   'min_samples_split': 2, 'n_estimators': 90}
    rf = RandomForestClassifier(**best_params, random_state=42)
    rf.fit(X_final, y)

    # === Step 3: Calibrate ===
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=5)
    rf_calibrated.fit(X_final, y)

    # === Step 4: Save model, tfidf, and column order ===
    save_model(rf_calibrated, "rf_calibrated.joblib")
    save_model(tfidf_pipeline, "tfidf_pipeline.joblib")
    save_model(train_columns, "train_columns.joblib")

    return rf_calibrated, tfidf_pipeline, train_columns


def plot_precision_recall(y_true, y_proba, target_precision=None, target_recall=None):
    """Plot Precision-Recall curve and return threshold"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()

    # Choose threshold closest to requested precision/recall
    if target_precision or target_recall:
        best_idx = 0
        best_score = -1
        for i, t in enumerate(thresholds):
            score = 0
            if target_precision:
                score -= abs(precision[i] - target_precision)
            if target_recall:
                score -= abs(recall[i] - target_recall)
            if score > best_score:
                best_idx, best_score = i, score
        return thresholds[best_idx]
    return 0.5  # default


def evaluate_at_threshold(threshold, y_true, y_proba):
    """Evaluate predictions at chosen threshold"""
    y_pred = (y_proba >= threshold).astype(int)
    print(f"Chosen threshold: {threshold:.3f}")
    print(classification_report(y_true, y_pred))


def evaluate_model(rf_calibrated, df_test, target_column="criticite"):
    """Evaluate trained model on test data"""
    X_test = df_test.drop(columns=[target_column])
    y_true = df_test[target_column]

    y_proba = rf_calibrated.predict_proba(X_test)[:, 1]
    chosen_thresh = plot_precision_recall(
        y_true, y_proba, target_precision=0.91, target_recall=0.83
    )
    evaluate_at_threshold(chosen_thresh, y_true, y_proba)
    return chosen_thresh


"""
if __name__ == "__main__":
    df_train = pd.read_csv("data/train.csv")  # replace with your file
    df_test = pd.read_csv("data/test.csv")    # replace with your file
    model, pipeline, _, _ = train_model(df_train)
    evaluate_model(model, df_test, pipeline)
    print("done")
"""

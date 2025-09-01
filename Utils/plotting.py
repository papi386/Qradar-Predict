from utils.plotting import plot_precision_recall, evaluate_at_threshold
import pandas as pd

y_true = pd.read_csv("data/recent_results.csv")['critical']
y_proba = pd.read_csv("data/recent_results.csv")['probability']

new_thresh = plot_precision_recall(y_true, y_proba, target_precision=0.91, target_recall=0.83)
evaluate_at_threshold(new_thresh, y_true, y_proba)

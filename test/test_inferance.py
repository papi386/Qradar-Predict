import pandas as pd
from model.save_load import load_model
from model.inferance import predict_new_data


if __name__ == "__main__":
    # Load trained model + pipeline
    rf_calibrated = load_model("rf_calibrated.joblib")
    tfidf_pipeline = load_model("tfidf_pipeline.joblib")

    # Example new QRadar-like events
    new_data = pd.DataFrame([
        {"event_info": "Suspicious login from multiple countries", "num_ips": 5, "unique_ip_countries": 3,
         "has_multiple_ip_ranges": 1},
        {"event_info": "Single suspicious login detected", "num_ips": 1, "unique_ip_countries": 1,
         "has_multiple_ip_ranges": 0},
        {"event_info": "Brute force attack detected", "num_ips": 10, "unique_ip_countries": 2,
         "has_multiple_ip_ranges": 1},
        {"event_info": "Malware download from suspicious URL", "num_ips": 2, "unique_ip_countries": 1,
         "has_multiple_ip_ranges": 0},
        {"event_info": "Multiple failed login attempts", "num_ips": 4, "unique_ip_countries": 1,
         "has_multiple_ip_ranges": 0},
        {"event_info": "New domain registered with suspicious pattern", "num_ips": 3, "unique_ip_countries": 2,
         "has_multiple_ip_ranges": 0},
        {"event_info": "Unauthorized access to sensitive file", "num_ips": 1, "unique_ip_countries": 1,
         "has_multiple_ip_ranges": 0},
        {"event_info": "Phishing email detected with URL", "num_ips": 2, "unique_ip_countries": 2,
         "has_multiple_ip_ranges": 0},
        {"event_info": "Botnet traffic detected from multiple IPs", "num_ips": 8, "unique_ip_countries": 4,
         "has_multiple_ip_ranges": 1},
        {"event_info": "Ransomware activity detected on endpoint", "num_ips": 1, "unique_ip_countries": 1,
         "has_multiple_ip_ranges": 0}
    ])

    # Predict
    results = predict_new_data(new_data, threshold=0.574)
    print("✅ Prediction completed. Results:")
    print(results[["prediction", "probability"]])
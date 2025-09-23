from model.inferance import predict_new_data
from siem.data_fetching import fetch_recent_offense_ids
from siem.data_preparation import extract_features_from_off
from siem.offense_investigation import investigate_offense

if __name__ == "__main__":
    API_TOKEN = "API TOKEN"
    QRADAR_HOST="QRADAR HOST"
    # Example new QRadar-like events
    ids = fetch_recent_offense_ids(QRADAR_HOST, API_TOKEN)
    print(len(ids))
    res = []
    for id in ids:
        offense = investigate_offense(id)
        res.append(offense)
    print(res)
    print("**********************")
    offenses = extract_features_from_off(res)
    print(offenses)

    # Predict
    results = predict_new_data(offenses, threshold=0.574)
    print("✅ Prediction completed. Results:")
    print(results[["prediction", "probability"]])
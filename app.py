import json
import threading
import time
import pandas as pd
from flask import Flask, render_template, Response

from model.inferance import predict_new_data
from siem.data_fetching import fetch_recent_offense_ids
from siem.data_preparation import extract_features_from_off
from siem.offense_investigation import investigate_offense

app = Flask(__name__)

API_TOKEN = "API TOKEN"
QRADAR_IP = "QRADAR IP"

critical_offenses = []
last_update = None  # track updates for SSE


def job_fetch_and_predict():
    global critical_offenses, last_update
    while True:
        try:
            ids = fetch_recent_offense_ids(QRADAR_IP, API_TOKEN, hours=3)
            res = []
            for oid in ids:
                offense = investigate_offense(oid)
                res.append(offense)

            offenses = extract_features_from_off(res)
            results = predict_new_data(offenses, threshold=0.574)

            ids_df = pd.DataFrame({"id": ids})

            # merge everything
            merged = pd.concat(
                [ids_df.reset_index(drop=True), offenses.reset_index(drop=True),
                 results[["prediction", "probability"]].reset_index(drop=True)],
                axis=1
            )

            # ✅ Keep only critical
            critical_only = merged[merged["prediction"] == 1]

            # ✅ Explicitly select the fields we want to stream
            critical_offenses = critical_only[["id", "event_info", "prediction", "probability"]].to_dict(orient="records")

            last_update = time.time()

            print(f"✅ Refreshed at {time.ctime()}, found {len(critical_offenses)} critical offenses")
        except Exception as e:
            print("⚠️ Error in job:", e)

        time.sleep(3 * 3600)


@app.route("/")
def index():
    # Now the HTML doesn’t need Flask data; it streams via SSE
    return render_template("critical.html")


@app.route("/stream")
def stream():
    def eventStream():
        last_sent = None
        while True:
            global critical_offenses, last_update
            data = json.dumps(critical_offenses)

            # push only if new offenses detected
            if data != last_sent:
                yield f"data: {data}\n\n"
                last_sent = data

            time.sleep(5)  # check interval

    return Response(eventStream(), mimetype="text/event-stream")


if __name__ == "__main__":
    threading.Thread(target=job_fetch_and_predict, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
# QRadar Real-Time Offense Prediction

This project connects to **IBM QRadar SIEM** via its REST API to:
- Fetch new offense details in **real-time (every 3 hours)**
- Collect related events and flows
- Extract features (domains, IPs, URLs, entropy, etc.)
- Pass the data directly into a **Machine Learning model**
- Predict which offenses are **critical** and display them ordered by criticity

---

## 🚀 Features
- Scheduled fetching of **new offenses every 3 hours**
- Runs **AQL queries** to retrieve related events and flows
- Feature engineering for each offense:
  - `num_domains`
  - `num_ips`
  - `num_urls`
  - `avg_domain_entropy`
  - `avg_length_of_urls`
  - `unique_ip_countries`
  - `has_multiple_ip_ranges`
- Direct integration with your ML model for **real-time prediction**
- Outputs ranked offenses (most critical first)

## 📦 Requirements
Install dependencies:
  bash
-pip install -r requirements.txt



## ⚙️ Usage

1. **Configure connection**  
   Edit the script and set your QRadar details:
     python
   QRADAR_HOST = "https://<qradar-host>"
   API_TOKEN = "PUT-YOUR-TOKEN"

2. **Run the script test_train to load the model**


3. **Run the script app.py**
 





import pandas as pd
from pipeline.pipeline import apply_pipeline

if __name__ == "__main__":
    # Example new data
    df = pd.DataFrame([
        {
            "event_info": "Suspicious login attempt detected",
            "num_ips": 2,
            "unique_ip_countries": 1,
            "has_multiple_ip_ranges": 0
        }
    ])

    # Apply pipeline
    df_final = apply_pipeline(df)

    print("✅ Pipeline applied. Transformed data sample:")
    print(df_final.head())

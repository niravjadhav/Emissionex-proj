
import json
import pandas as pd

with open("outputs/forensic_incidents.json", encoding="utf-8") as f:
    incidents = json.load(f)

df = pd.read_csv("outputs/shap_incident_explanations.csv")

assert len(incidents) == 443, f"Unexpected incident count: {len(incidents)}"
assert df["incident_id"].nunique() == len(incidents)
assert len(df) == len(incidents) * 3

for incident in incidents:
    features = incident["top_shap_features"]
    energy = incident["energy_kwh"]

    assert len(features) == 3
    assert abs(
        energy["actual"] - energy["expected"] - energy["deviation"]
    ) < 1e-5

    assert all(
        features[i]["absolute_shap"] >= features[i + 1]["absolute_shap"]
        for i in range(2)
    )

print("Basic validation passed.")
print("Incidents:", len(incidents))
print("Feature explanations:", len(df))

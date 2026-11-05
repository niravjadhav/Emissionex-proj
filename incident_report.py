
import json
from pathlib import Path

source = Path("outputs/forensic_incidents.json")
destination = Path("outputs/incident_report.txt")

with source.open("r", encoding="utf-8") as f:
    incidents = json.load(f)

with destination.open("w", encoding="utf-8") as f:
    f.write("EMISSIONEX — FORENSIC INCIDENT REPORT\n")
    f.write("=" * 50 + "\n\n")

    for incident in incidents:
        location = incident.get("location", {})
        energy = incident.get("energy_kwh", {})
        features = incident.get("top_shap_features", [])

        f.write(f"Incident ID: {incident.get('incident_id', 'Unknown')}\n")
        f.write(f"Timestamp: {incident.get('timestamp', 'Unknown')}\n")
        f.write(f"Severity: {incident.get('severity', 'Unknown')}\n")

        f.write("\nLOCATION\n")
        f.write(f"Plant: {location.get('plant_id', 'Unknown')}\n")
        f.write(f"Building: {location.get('building_id', 'Unknown')}\n")
        f.write(f"Department: {location.get('department_id', 'Unknown')}\n")
        f.write(f"Machine: {location.get('machine_id', 'Unknown')}\n")
        f.write(f"Machine type: {location.get('machine_type', 'Unknown')}\n")

        f.write("\nENERGY ANALYSIS\n")
        f.write(f"Actual energy: {energy.get('actual', 'Unknown')} kWh\n")
        f.write(f"Expected energy: {energy.get('expected', 'Unknown')} kWh\n")
        f.write(f"Energy deviation: {energy.get('deviation', 'Unknown')} kWh\n")
        f.write(f"Excess CO2e: {incident.get('excess_co2e_kg', 'Unknown')} kg\n")
        f.write(f"Model: {incident.get('baseline_model_id', 'Unknown')}\n")

        f.write("\nTOP SHAP FEATURES\n")
        for feature in features:
            f.write(
                f"{feature.get('rank', '?')}. "
                f"{feature.get('feature', 'Unknown')} | "
                f"Value: {feature.get('feature_value', 'Unknown')} | "
                f"SHAP: {feature.get('shap_value', 'Unknown')}\n"
            )

        f.write("\n" + "-" * 50 + "\n\n")

print("Forensic report generated successfully.")
print("Incidents included:", len(incidents))
print("Saved to:", destination)

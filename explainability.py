"""Explain detected EmissioNex incidents with SHAP.

Uses the exact saved baseline model that scored each incident. Ground-truth
fault labels are never passed to the model as features.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from baseline import encode_features, load_baseline_artifacts

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
INCIDENTS_PATH = OUTPUTS / "detected_incidents.csv"
MODELS_PATH = OUTPUTS / "baseline_models"


def _json_value(value):
    """Convert pandas/numpy values to JSON-safe Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    started = time.perf_counter()
    incidents = pd.read_csv(INCIDENTS_PATH)
    if incidents.empty:
        raise ValueError(f"No incidents found in {INCIDENTS_PATH}")
    if "baseline_model_id" not in incidents:
        raise ValueError("Incidents file is missing baseline_model_id")

    artifacts = load_baseline_artifacts(MODELS_PATH)
    explanation_rows: list[dict] = []
    incident_reports: list[dict] = []
    all_shap_rows: list[pd.DataFrame] = []
    skipped: list[str] = []
    explainers: dict = {}

    print(f"Incidents: {len(incidents)}")
    print(f"Available models: {list(artifacts.models)}")

    for model_id_value, group in incidents.groupby("baseline_model_id", sort=False, dropna=False):
        model_id = str(model_id_value)
        if model_id not in artifacts.models:
            skipped.extend(group["incident_id"].astype(str).tolist())
            print(f"Skipping {len(group)} incidents: saved model '{model_id}' not found")
            continue

        model = artifacts.models[model_id]
        X = encode_features(
            group,
            artifacts.categorical,
            artifacts.numeric,
            artifacts.feature_columns,
        )
        if model_id not in explainers:
            explainers[model_id] = shap.TreeExplainer(model)
        shap_values = np.asarray(explainers[model_id].shap_values(X))
        if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
            shap_values = shap_values[:, :, 0]
        if shap_values.shape != X.shape:
            raise ValueError(
                f"SHAP returned {shap_values.shape}, but feature matrix is {X.shape} "
                f"for model {model_id}"
            )

        all_shap_rows.append(pd.DataFrame(np.abs(shap_values), columns=X.columns))

        for position, (_, row) in enumerate(group.iterrows()):
            ranking = pd.DataFrame({
                "feature": X.columns,
                "feature_value": X.iloc[position].to_numpy(),
                "shap_value": shap_values[position],
            })
            ranking["absolute_shap"] = ranking["shap_value"].abs()
            top3 = ranking.nlargest(3, "absolute_shap")
            top_features = []

            for rank, (_, feature_row) in enumerate(top3.iterrows(), start=1):
                record = {
                    "incident_id": row.get("incident_id"),
                    "source_row_id": row.get("source_row_id"),
                    "timestamp": row.get("timestamp"),
                    "plant_id": row.get("plant_id"),
                    "building_id": row.get("building_id"),
                    "department_id": row.get("department_id"),
                    "machine_id": row.get("machine_id"),
                    "machine_type": row.get("machine_type"),
                    "severity": row.get("severity"),
                    "actual_kWh": row.get("kWh"),
                    "expected_kWh": row.get("expected_kWh"),
                    "residual_kWh": row.get("residual_kWh"),
                    "excess_co2e_kg": row.get("excess_co2e_kg"),
                    "baseline_model_id": model_id,
                    "feature_rank": rank,
                    "feature": feature_row["feature"],
                    "feature_value": feature_row["feature_value"],
                    "shap_value": feature_row["shap_value"],
                    "absolute_shap": feature_row["absolute_shap"],
                }
                explanation_rows.append(record)
                top_features.append({
                    "rank": rank,
                    "feature": feature_row["feature"],
                    "feature_value": _json_value(feature_row["feature_value"]),
                    "shap_value": _json_value(feature_row["shap_value"]),
                    "absolute_shap": _json_value(feature_row["absolute_shap"]),
                })

            incident_reports.append({
                "incident_id": _json_value(row.get("incident_id")),
                "timestamp": _json_value(row.get("timestamp")),
                "location": {
                    "plant_id": _json_value(row.get("plant_id")),
                    "building_id": _json_value(row.get("building_id")),
                    "department_id": _json_value(row.get("department_id")),
                    "machine_id": _json_value(row.get("machine_id")),
                    "machine_type": _json_value(row.get("machine_type")),
                },
                "severity": _json_value(row.get("severity")),
                "energy_kwh": {
                    "actual": _json_value(row.get("kWh")),
                    "expected": _json_value(row.get("expected_kWh")),
                    "deviation": _json_value(row.get("residual_kWh")),
                },
                "excess_co2e_kg": _json_value(row.get("excess_co2e_kg")),
                "baseline_model_id": model_id,
                "top_shap_features": top_features,
                "interpretation_note": (
                    "SHAP values explain contributions to the expected-energy prediction; "
                    "they do not prove a physical fault or causal mechanism."
                ),
            })
        print(f"Explained {len(group)} incidents with {model_id}")

    if not explanation_rows:
        raise RuntimeError("No explanations were generated; check the model files and incident data.")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    explanations = pd.DataFrame(explanation_rows)
    csv_path = OUTPUTS / "shap_incident_explanations.csv"
    json_path = OUTPUTS / "forensic_incidents.json"
    chart_path = OUTPUTS / "shap_summary.png"
    explanations.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(incident_reports, file, indent=2, ensure_ascii=False)

    # Aggregate mean absolute SHAP over every feature and every explained incident.
    full_values = pd.concat(all_shap_rows, ignore_index=True)
    importance = full_values.mean(axis=0).sort_values(ascending=True).tail(12)
    plt.figure(figsize=(10, 6))
    importance.plot(kind="barh")
    plt.title("EmissioNex: Features Influencing Expected Energy")
    plt.xlabel("Mean absolute SHAP value (kWh prediction scale)")
    plt.ylabel("Encoded feature")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close()

    if skipped:
        pd.DataFrame({"incident_id": skipped}).to_csv(
            OUTPUTS / "shap_skipped_incidents.csv", index=False
        )

    print("\n=== Explainability complete ===")
    print(f"Input incidents: {len(incidents)}")
    print(f"Successfully explained: {len(incident_reports)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Feature explanation rows: {len(explanations)}")
    print(f"CSV: {csv_path.relative_to(ROOT)}")
    print(f"JSON: {json_path.relative_to(ROOT)}")
    print(f"Chart: {chart_path.relative_to(ROOT)}")
    print(f"Runtime: {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()

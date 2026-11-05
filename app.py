from pathlib import Path
import json
import sqlite3
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DB_PATH = ROOT / "emissionex_workflow.db"

st.set_page_config(
    page_title="EmissioNex | Operations",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
  --green-950:#12372b; --green-900:#174a37; --green-800:#1f6446;
  --green-700:#258454; --green-600:#2e9b62; --green-100:#e5f5eb;
  --green-50:#f4fbf6; --ink:#18352a; --muted:#688074; --line:#dcebe1;
}
.stApp { background: linear-gradient(180deg,#e6f5e8 0%,#edf8ef 48%,#e1f1e5 100%); color:var(--ink); }
.block-container {max-width: 1560px; padding-top: 1.35rem; padding-bottom: 2.5rem;}
[data-testid="stHeader"] {background:rgba(230,245,232,.9);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#d9efdf 0%,#eaf6ed 100%); border-right:1px solid #dcebe1;}
[data-testid="stSidebar"] > div {padding-top:1.1rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {color:#174a37;}
[data-testid="stMetric"] {
  background:linear-gradient(145deg,#ffffff 0%,#eaf7ed 100%);
  border:1px solid #d8ebdd; padding:17px 18px; border-radius:17px;
  box-shadow:0 3px 12px rgba(27,84,53,.045); min-height:112px;
}
[data-testid="stMetricLabel"] {font-size:.83rem; color:#648072; font-weight:600;}
[data-testid="stMetricValue"] {font-size:1.72rem; color:#174a37; font-weight:750;}
[data-testid="stMetricDelta"] {font-size:.8rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius:17px !important; border-color:#dcebe1 !important;
  background:rgba(250,255,251,.94); box-shadow:0 4px 15px rgba(31,83,54,.045);
}
div.stButton > button, div.stDownloadButton > button {
  border-radius:11px; font-weight:650; min-height:2.65rem;
  border:1px solid #cfe4d5; transition:all .15s ease;
}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
  background:linear-gradient(135deg,#258454,#1d6d45); color:white; border:0;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  border-color:#258454; color:#1d6d45; box-shadow:0 3px 10px rgba(37,132,84,.12);
}
div.stButton > button[kind="primary"]:hover {color:white; background:#174a37;}
[data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label,
[data-testid="stSlider"] label, [data-testid="stDateInput"] label {color:#365b47;font-weight:600;}
input, textarea, [data-baseweb="select"] > div {border-radius:10px !important;}
h1,h2,h3 {color:#174a37; letter-spacing:-.025em;}
h2 {margin-top:.35rem !important; margin-bottom:.25rem !important;}
hr {border-color:#dcebe1;}
.hero {
  padding:25px 28px; border-radius:22px; margin-bottom:21px;
  background:radial-gradient(circle at 90% 10%,rgba(149,230,174,.28),transparent 28%),
             linear-gradient(115deg,#12372b 0%,#1d6946 58%,#2e9b62 100%);
  color:white; box-shadow:0 10px 28px rgba(23,74,55,.15);
  position:relative; overflow:hidden;
}
.hero h1 {color:white; margin:0; font-size:2.25rem; letter-spacing:-.04em;}
.hero p {color:#e1f6e7; margin:.45rem 0 0 0; font-size:1rem; max-width:800px;}
.kicker {text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:800; color:#a8efbd; margin-bottom:7px;}
.section-intro {color:#688074; margin-top:0; margin-bottom:1.2rem; font-size:.95rem;}
.eyebrow {font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; color:#648072; font-weight:750;}
div[data-testid="stProgress"] > div > div {background:linear-gradient(90deg,#258454,#62c987);}
[data-testid="stAlert"] {border-radius:13px;}
[data-testid="stDataFrame"], [data-testid="stTable"] {border-radius:12px; overflow:hidden;}
[data-testid="stRadio"] label {font-weight:600;}
[data-testid="stPlotlyChart"] {background:#fafffb;border:1px solid #d1e8d7;border-radius:16px;padding:8px;}
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
@media (max-width: 800px) {
  .block-container {padding-left:1rem;padding-right:1rem;}
  .hero {padding:20px;border-radius:17px;}
  .hero h1 {font-size:1.8rem;}
}
</style>
""", unsafe_allow_html=True)


def read_csv(name):
    p = OUT / name
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def read_json_records(name):
    p = OUT / name
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            for key in ("incidents", "recommendations", "records", "results", "data"):
                if isinstance(raw.get(key), list):
                    return [x for x in raw[key] if isinstance(x, dict)]
            return [raw]
    except Exception:
        return []
    return []


def flatten_incident(obj):
    row = dict(obj)
    loc = row.pop("location", None)
    if isinstance(loc, dict):
        for k, v in loc.items():
            row.setdefault(k, v)
    energy = row.pop("energy_kwh", None)
    if isinstance(energy, dict):
        for k, v in energy.items():
            row.setdefault("energy_" + k, v)
    return row


def first_col(df, candidates):
    return next((c for c in candidates if c in df.columns), None)


def fmt(v, digits=1):
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def norm_severity(v):
    s = str(v or "unknown").strip().lower()
    if s in ("critical", "high"):
        return "High"
    if s in ("medium", "moderate"):
        return "Medium"
    if s == "low":
        return "Low"
    return "Unknown"


def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""
        CREATE TABLE IF NOT EXISTS incident_workflow (
            incident_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'Open',
            owner TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        )
    """)
    con.commit()
    return con


def load_workflow():
    with db_connect() as con:
        return pd.read_sql_query("SELECT * FROM incident_workflow", con)


def save_workflow(incident_id, status, owner, notes):
    with db_connect() as con:
        con.execute("""
            INSERT INTO incident_workflow (incident_id, status, owner, notes, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(incident_id) DO UPDATE SET
              status=excluded.status, owner=excluded.owner,
              notes=excluded.notes, updated_at=excluded.updated_at
        """, (incident_id, status, owner, notes, datetime.now().isoformat(timespec="seconds")))
        con.commit()


def navigate_to_incident(incident_id):
    st.session_state["selected_incident_id"] = incident_id
    st.session_state["page"] = "Incident Workspace"


if not OUT.exists():
    st.error(f"Can't find the outputs folder: {OUT}")
    st.info("Keep app.py in the EmissioNex project root, beside the outputs folder.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_project_data():
    forensic = read_json_records("forensic_incidents.json")
    if forensic:
        incidents = pd.DataFrame([flatten_incident(x) for x in forensic])
    else:
        incidents = read_csv("detected_incidents.csv")
        if incidents.empty:
            incidents = read_csv("detection_results.csv")
    recommendations = read_json_records("recommendations.json")
    shap = read_csv("shap_incident_explanations.csv")
    localization = read_csv("localization_summary.csv")
    return incidents, recommendations, shap, localization


incidents, recommendation_records, shap_df, localization_df = load_project_data()
if incidents.empty:
    st.error("No incidents were found in outputs/forensic_incidents.json or outputs/detected_incidents.csv.")
    st.stop()

id_col = first_col(incidents, ["incident_id", "id", "incident"])
time_col = first_col(incidents, ["timestamp", "datetime", "time", "date"])
machine_col = first_col(incidents, ["machine_id", "machine", "asset_id"])
type_col = first_col(incidents, ["machine_type", "equipment_type"])
severity_col = first_col(incidents, ["severity", "risk_level"])
actual_col = first_col(incidents, ["energy_actual", "actual_kwh", "actual_energy_kwh", "actual"])
expected_col = first_col(incidents, ["energy_expected", "expected_kwh", "expected_energy_kwh", "expected"])
deviation_col = first_col(incidents, ["energy_deviation", "deviation_kwh", "deviation"])
co2_col = first_col(incidents, ["excess_co2e_kg", "excess_co2_kg", "co2e_excess_kg"])

if id_col is None:
    incidents["incident_id"] = [f"INC-{i+1:04d}" for i in range(len(incidents))]
    id_col = "incident_id"
if machine_col is None:
    incidents["machine_id"] = "Unknown"
    machine_col = "machine_id"
if type_col is None:
    incidents["machine_type"] = "Equipment"
    type_col = "machine_type"
if severity_col is None:
    incidents["severity"] = "Unknown"
    severity_col = "severity"

for c in [actual_col, expected_col, deviation_col, co2_col]:
    if c:
        incidents[c] = pd.to_numeric(incidents[c], errors="coerce")
if time_col:
    incidents[time_col] = pd.to_datetime(incidents[time_col], errors="coerce")

incidents["_id"] = incidents[id_col].astype(str)
incidents["_severity"] = incidents[severity_col].map(norm_severity)
incidents["_actual"] = pd.to_numeric(incidents[actual_col], errors="coerce") if actual_col else float("nan")
incidents["_expected"] = pd.to_numeric(incidents[expected_col], errors="coerce") if expected_col else float("nan")
incidents["_deviation"] = pd.to_numeric(incidents[deviation_col], errors="coerce") if deviation_col else incidents["_actual"] - incidents["_expected"]
incidents["_co2"] = pd.to_numeric(incidents[co2_col], errors="coerce") if co2_col else float("nan")

rec_by_id = {
    str(r.get("incident_id")): r
    for r in recommendation_records
    if r.get("incident_id") is not None
}

workflow = load_workflow()
workflow_by_id = {str(r["incident_id"]): r for _, r in workflow.iterrows()}
workflow_map = {
    iid: workflow_by_id.get(iid, {"status": "Open", "owner": "", "notes": "", "updated_at": ""})
    for iid in incidents["_id"]
}
incidents["_status"] = incidents["_id"].map(lambda x: workflow_map[x]["status"])
incidents["_owner"] = incidents["_id"].map(lambda x: workflow_map[x]["owner"])
incidents["_urgency_rank"] = incidents["_severity"].map({"High": 0, "Medium": 1, "Low": 2, "Unknown": 3})

st.markdown("""
<div class="hero">
  <div class="kicker">Factory operations · Carbon intelligence</div>
  <h1>EmissioNex</h1>
  <p>Find the biggest energy exceptions, decide what to investigate, and track actions through to resolution.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Work queue")
    page = st.radio(
        "Go to",
        ["Action Center", "Incident Workspace", "Performance Overview", "Factory Savings & Impact", "Technical Evidence"],
        label_visibility="collapsed",
        key="page",
    )
    st.divider()
    st.markdown("### Quick find")
    search_term = st.text_input(
        "Incident or equipment",
        placeholder="Search ID, machine, equipment type…",
        help="Filters the incident queue and workspace selector.",
    ).strip().lower()
    st.markdown("### Narrow the queue")
    severity_choices = [s for s in ["High", "Medium", "Low", "Unknown"] if s in incidents["_severity"].unique()]
    chosen_severity = st.multiselect("Urgency", severity_choices, default=severity_choices)
    status_choices = ["Open", "Under Investigation", "Resolved", "Accepted Risk"]
    chosen_status = st.multiselect("Workflow status", status_choices, default=status_choices)
    st.divider()
    st.markdown("### Cost assumptions")
    tariff = st.number_input(
        "Electricity tariff (₹/kWh)",
        min_value=0.0,
        max_value=1000.0,
        value=8.0,
        step=0.5,
        help="Enter your factory's applicable electricity rate. The default ₹8/kWh is only an editable example, not a verified tariff.",
    )
    recovery_pct = st.slider(
        "Potential excess energy to recover (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Scenario assumption only. It is not a prediction or verified saving.",
    )
    machine_choices = sorted(incidents[machine_col].dropna().astype(str).unique().tolist())
    chosen_machines = st.multiselect("Equipment", machine_choices)
    if time_col and incidents[time_col].notna().any():
        start = incidents[time_col].min().date()
        end = incidents[time_col].max().date()
        date_range = st.date_input("Incident period", value=(start, end))
    else:
        date_range = None
    st.divider()
    st.caption("Status, owner and notes are saved locally in emissionex_workflow.db.")


view = incidents[incidents["_severity"].isin(chosen_severity) & incidents["_status"].isin(chosen_status)].copy()
if search_term:
    searchable = view["_id"].astype(str).str.lower()
    searchable = searchable.str.cat(view[machine_col].astype(str).str.lower(), sep=" ")
    searchable = searchable.str.cat(view[type_col].astype(str).str.lower(), sep=" ")
    view = view[searchable.str.contains(re.escape(search_term), na=False)]
if chosen_machines:
    view = view[view[machine_col].astype(str).isin(chosen_machines)]
if date_range and time_col and isinstance(date_range, tuple) and len(date_range) == 2:
    view = view[view[time_col].dt.date.between(date_range[0], date_range[1])]

priority_order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
view["_urgency_rank"] = view["_severity"].map(priority_order)
view = view.sort_values(["_urgency_rank", "_co2"], ascending=[True, False], na_position="last")
open_count = int((incidents["_status"] == "Open").sum())
investigating_count = int((incidents["_status"] == "Under Investigation").sum())
resolved_count = int((incidents["_status"] == "Resolved").sum())
high_count = int((incidents["_severity"] == "High").sum())
co2_total = incidents["_co2"].sum(min_count=1)

if page == "Action Center":
    st.markdown("## Action Center")
    st.markdown('<p class="section-intro">Your daily control room: prioritize exceptions, assign follow-up, and track progress from one place.</p>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Flagged incidents", f"{len(incidents):,}")
    b.metric("High urgency", f"{high_count:,}")
    c.metric("Still open", f"{open_count:,}")
    d.metric("Estimated excess CO₂e", f"{fmt(co2_total)} kg")

    st.markdown("### Priority queue")
    st.caption("Showing the highest-priority matching incidents first. Select “Open investigation” to review evidence and record an action. CO₂e values are estimates.")
    if view.empty:
        st.success("No incidents match these filters. Clear or widen the sidebar filters to see more.")
    else:
        queue = view.head(30)
        for _, row in queue.iterrows():
            iid = row["_id"]
            rec = rec_by_id.get(iid, {})
            actions = rec.get("recommended_actions", []) if isinstance(rec, dict) else []
            first_action = ""
            if actions:
                first = actions[0]
                first_action = first.get("action", "") if isinstance(first, dict) else str(first)
            with st.container(border=True):
                left, mid, right = st.columns([4, 1.2, 1.5])
                with left:
                    st.markdown(f"**{iid}**")
                    stamp = row[time_col].strftime("%d %b %Y · %H:%M") if time_col and pd.notna(row[time_col]) else "Time unavailable"
                    st.caption(f"{row[machine_col]} · {row[type_col]} · {stamp}")
                    st.write(first_action or "Open the incident workspace to review next steps.")
                with mid:
                    st.metric("Urgency", row["_severity"])
                    st.caption(f"Status: {row['_status']}")
                with right:
                    st.metric("Est. excess CO₂e", f"{fmt(row['_co2'])} kg")
                    st.button(
                        "Open investigation →",
                        key=f"open_{iid}",
                        use_container_width=True,
                        on_click=navigate_to_incident,
                        args=(iid,),
                    )
    st.markdown("### Workflow pulse")
    w1, w2, w3 = st.columns(3)
    w1.metric("Open", open_count)
    w2.metric("Under investigation", investigating_count)
    w3.metric("Resolved", resolved_count)
    st.progress(resolved_count / max(len(incidents), 1), text=f"{resolved_count} of {len(incidents)} incidents marked resolved")

elif page == "Incident Workspace":
    st.markdown("## Incident Workspace")
    st.markdown('<p class="section-intro">Review the evidence, follow recommended checks, and keep ownership and resolution notes together.</p>', unsafe_allow_html=True)
    ids = view["_id"].tolist() or incidents["_id"].tolist()
    remembered = st.session_state.get("selected_incident_id")
    if remembered not in ids:
        remembered = ids[0] if ids else incidents["_id"].iloc[0]
    chosen_id = st.selectbox("Choose incident", ids, index=ids.index(remembered), format_func=lambda x: f"{x} · {incidents.loc[incidents['_id'] == x, machine_col].iloc[0]}")
    row = incidents[incidents["_id"] == chosen_id].iloc[0]
    rec = rec_by_id.get(chosen_id, {})
    wf = workflow_map.get(chosen_id, {"status": "Open", "owner": "", "notes": "", "updated_at": ""})
    stamp = row[time_col].strftime("%d %b %Y, %H:%M") if time_col and pd.notna(row[time_col]) else "Unknown time"
    st.markdown(f"### {chosen_id}")
    st.caption(f"{row[machine_col]} · {row[type_col]} · Detected {stamp}")
    st.markdown(f"**Urgency: {row['_severity']}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual energy", f"{fmt(row['_actual'])} kWh")
    m2.metric("Expected energy", f"{fmt(row['_expected'])} kWh")
    m3.metric("Above baseline", f"{fmt(row['_deviation'])} kWh")
    m4.metric("Estimated excess CO₂e", f"{fmt(row['_co2'])} kg")
    if pd.notna(row["_actual"]) and pd.notna(row["_expected"]):
        if row["_actual"] > row["_expected"]:
            delta_pct = (row["_actual"] - row["_expected"]) / row["_expected"] * 100 if row["_expected"] else None
            pct_msg = f" ({fmt(delta_pct)}% above expected)" if delta_pct is not None else ""
            st.warning(f"**What this means:** recorded energy is {fmt(row['_deviation'])} kWh higher than the baseline{pct_msg}. This is an investigation signal, not proof of a fault.")
        else:
            st.info("Recorded energy is not above the baseline for this record. Review the incident details to understand why it was flagged.")
    st.markdown("### Recommended next steps")
    actions = rec.get("recommended_actions", []) if isinstance(rec, dict) else []
    if actions:
        mode = rec.get("generation_mode", "unknown")
        source_labels = {
        "nugen_llm_grounded": "Nugen AI · evidence-grounded",
        "ollama_llm_grounded": "Local Ollama · evidence-grounded",
         "rule_based_fallback": "Rule-based fallback",
         }
        st.caption("Suggestion source: " + source_labels.get(mode, mode.replace("_", " ")))
        for i, action in enumerate(actions, 1):
            if not isinstance(action, dict):
                action = {"action": str(action)}
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**Step {i}: {action.get('action', 'Review incident')}**")
                c2.caption(f"Priority: {action.get('priority', 'Review')}")
                if action.get("reason"):
                    c1.write(f"**Why:** {action['reason']}")
                if action.get("evidence_basis"):
                    c1.caption(f"Evidence basis: {action['evidence_basis']}")
                if action.get("evidence_ids"):
                    c1.caption("Evidence IDs: " + ", ".join(map(str, action["evidence_ids"])))
    else:
        st.warning("No saved recommendation was found for this incident. Re-run the recommendation engine for matching incident IDs.")
    evidence = rec.get("retrieved_evidence", []) if isinstance(rec, dict) else []
    if evidence:
        with st.expander("View retrieved source passages"):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                st.markdown(f"**{item.get('evidence_id', 'Source')} · {item.get('document', 'Document')}**")
                if item.get("section"):
                    st.caption(str(item["section"]))
                st.write(item.get("excerpt", "No excerpt stored."))
                st.divider()

    st.markdown("### Assign and track this incident")
    with st.form(f"workflow_form_{chosen_id}"):
        statuses = ["Open", "Under Investigation", "Resolved", "Accepted Risk"]
        current_status = wf.get("status", "Open")
        status_idx = statuses.index(current_status) if current_status in statuses else 0
        new_status = st.selectbox("Status", statuses, index=status_idx)
        new_owner = st.text_input("Assigned to", value=str(wf.get("owner", "") or ""), placeholder="e.g., Maintenance team / Engineer name")
        new_notes = st.text_area("Investigation notes / verified cause / action taken", value=str(wf.get("notes", "") or ""), placeholder="Record what the engineer checked, what was found, and what action was taken.", height=120)
        submitted = st.form_submit_button("Save incident update", type="primary")
        if submitted:
            save_workflow(chosen_id, new_status, new_owner.strip(), new_notes.strip())
            st.success("Incident update saved locally.")
            st.rerun()
    if wf.get("updated_at"):
        st.caption(f"Last updated: {wf['updated_at']}")

elif page == "Performance Overview":
    st.markdown("## Performance Overview")
    st.markdown('<p class="section-intro">A quick read on incident patterns, estimated emissions impact, and investigation progress.</p>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Estimated excess CO₂e", f"{fmt(co2_total)} kg")
    b.metric("Equipment affected", f"{incidents[machine_col].nunique():,}")
    c.metric("Resolution rate", f"{resolved_count / max(len(incidents), 1) * 100:.1f}%")
    left, right = st.columns(2)
    with left:
        sev = incidents["_severity"].value_counts().rename_axis("Urgency").reset_index(name="Incidents")
        st.plotly_chart(px.bar(sev, x="Urgency", y="Incidents", color="Urgency", text="Incidents", title="Incidents by urgency", template="plotly_white", color_discrete_map={"High":"#dc6b58","Medium":"#e9b949","Low":"#2e9b62","Unknown":"#9aa9a0"}), use_container_width=True)
    with right:
        if incidents["_co2"].notna().any():
            m = incidents.groupby(machine_col)["_co2"].sum().sort_values(ascending=False).head(10).reset_index()
            m.columns = ["Equipment", "Estimated excess CO₂e (kg)"]
            st.plotly_chart(px.bar(m, x="Estimated excess CO₂e (kg)", y="Equipment", orientation="h", title="Equipment with highest estimated excess emissions", template="plotly_white", color_discrete_sequence=["#258454"]), use_container_width=True)
    if time_col and incidents[time_col].notna().any():
        trend = incidents.dropna(subset=[time_col]).set_index(time_col).resample("D")["_co2"].sum(min_count=1).reset_index()
        trend.columns = ["Date", "Estimated excess CO₂e (kg)"]
        st.plotly_chart(px.line(trend, x="Date", y="Estimated excess CO₂e (kg)", markers=True, title="Estimated excess emissions over time", template="plotly_white", color_discrete_sequence=["#258454"]), use_container_width=True)
    st.markdown("### Status by workflow")
    status_counts = incidents["_status"].value_counts().rename_axis("Status").reset_index(name="Incidents")
    st.plotly_chart(px.bar(status_counts, x="Status", y="Incidents", color="Status", text="Incidents", template="plotly_white", color_discrete_sequence=["#258454","#62b879","#9acdaa","#c4dfca"]), use_container_width=True)

elif page == "Factory Savings & Impact":
    st.markdown("## Factory Savings & Impact")
    st.markdown('<p class="section-intro">Explore what-if recovery scenarios using editable assumptions. These figures are planning estimates, not verified savings.</p>', unsafe_allow_html=True)
    positive_deviation = incidents["_deviation"].clip(lower=0)
    estimated_excess_kwh = positive_deviation.sum(min_count=1)
    potential_kwh = positive_deviation * (recovery_pct / 100.0)
    potential_cost = potential_kwh * tariff
    total_potential_cost = potential_cost.sum(min_count=1)
    full_cost_opportunity = estimated_excess_kwh * tariff if pd.notna(estimated_excess_kwh) else float("nan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive energy deviation", f"{fmt(estimated_excess_kwh)} kWh")
    c2.metric("Tariff assumption", f"₹{tariff:,.2f}/kWh")
    c3.metric(f"Scenario energy recovery ({recovery_pct}%)", f"{fmt(potential_kwh.sum(min_count=1))} kWh")
    c4.metric("Potential cost opportunity", f"₹{fmt(total_potential_cost)}")
    st.caption("The potential cost opportunity is calculated as positive energy deviation × assumed recovery percentage × electricity tariff. It is not actual money saved.")
    st.markdown("### Compare recovery scenarios")
    scenario_rows = []
    for pct in [0, 25, 50, 75, 100]:
        kwh = positive_deviation.sum(min_count=1) * pct / 100.0 if positive_deviation.notna().any() else float("nan")
        scenario_rows.append({
            "Assumed recovery": f"{pct}%",
            "Energy potentially recovered (kWh)": kwh,
            "Potential cost opportunity (₹)": kwh * tariff if pd.notna(kwh) else float("nan"),
        })
    scenario_df = pd.DataFrame(scenario_rows)
    l, r = st.columns(2)
    with l:
        st.plotly_chart(px.bar(scenario_df, x="Assumed recovery", y="Potential cost opportunity (₹)", title="Illustrative cost opportunity by scenario", text_auto=".0f", template="plotly_white", color_discrete_sequence=["#258454"]), use_container_width=True)
    with r:
        st.plotly_chart(px.bar(scenario_df, x="Assumed recovery", y="Energy potentially recovered (kWh)", title="Illustrative recoverable energy by scenario", text_auto=".0f", template="plotly_white", color_discrete_sequence=["#62b879"]), use_container_width=True)
    st.markdown("### Where to investigate first")
    machine_impact = incidents.copy()
    machine_impact["_positive_deviation"] = machine_impact["_deviation"].clip(lower=0)
    machine_impact = machine_impact.groupby(machine_col, dropna=False).agg(
        Incidents=("_id", "count"),
        **{"Positive deviation (kWh)": ("_positive_deviation", "sum")}
    ).reset_index().sort_values("Positive deviation (kWh)", ascending=False)
    machine_impact["Scenario cost opportunity (₹)"] = machine_impact["Positive deviation (kWh)"] * (recovery_pct / 100.0) * tariff
    machine_impact = machine_impact.rename(columns={machine_col: "Equipment"})
    st.dataframe(machine_impact.head(20), use_container_width=True, hide_index=True)
    st.download_button("Download factory impact analysis CSV", machine_impact.to_csv(index=False).encode("utf-8"), "factory_savings_impact.csv", "text/csv")
    st.warning("Before using this for financial decisions, confirm that energy deviation values are comparable kWh quantities, set the actual tariff from your bill, and validate recoverable savings after engineering review.")

else:
    st.markdown("## Technical Evidence")
    st.caption("For engineers and project reviewers. SHAP and localization are supporting signals, not definitive physical root-cause proof.")
    if not shap_df.empty:
        st.markdown("### SHAP incident explanations")
        shap_id_col = first_col(shap_df, ["incident_id", "id"])
        if shap_id_col:
            chosen = st.selectbox("Incident for SHAP detail", incidents["_id"].tolist(), key="technical_shap")
            matched = shap_df[shap_df[shap_id_col].astype(str) == chosen]
            st.dataframe(matched if not matched.empty else shap_df.head(100), use_container_width=True, hide_index=True)
        else:
            st.dataframe(shap_df.head(200), use_container_width=True, hide_index=True)
        st.download_button("Download SHAP CSV", shap_df.to_csv(index=False).encode("utf-8"), "shap_incident_explanations.csv", "text/csv")
    else:
        st.info("SHAP explanation file is not available.")
    if not localization_df.empty:
        st.markdown("### Source localization")
        st.dataframe(localization_df.head(300), use_container_width=True, hide_index=True)
        st.download_button("Download localization CSV", localization_df.to_csv(index=False).encode("utf-8"), "localization_summary.csv", "text/csv")
    else:
        st.info("Localization summary file is not available.")
    st.markdown("### Interpretation")
    st.write("Actual energy is the recorded consumption; expected energy is the baseline estimate; the difference is actual minus expected. Excess CO₂e is a project estimate. SHAP explains model behaviour; recommendations are investigation suggestions, not confirmed diagnoses.")

st.sidebar.divider()
st.sidebar.caption(f"{len(incidents):,} source incidents loaded")
st.sidebar.caption("Local workflow database: emissionex_workflow.db")

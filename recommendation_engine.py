"""Evidence-grounded recommendation generation for EmissioNex.

Pipeline: forensic incident -> semantic retrieval (ChromaDB) -> Nugen aligned-model inference by default, or optional local Ollama for testing
narrative grounded in retrieved passages -> safety/evidence checks -> JSON/CSV/report.
If inference is unavailable or rejected, conservative rule-based recommendations are emitted.
The active retrieval and generation modes are recorded in every output.
"""
from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_pipeline import LocalRAG

INPUT_PATH = Path("outputs/forensic_incidents.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_JSON = OUTPUT_DIR / "recommendations.json"
OUTPUT_CSV = OUTPUT_DIR / "recommendations.csv"
OUTPUT_REPORT = OUTPUT_DIR / "recommendation_report.txt"


def _build_query(incident: dict[str, Any]) -> str:
    location = incident.get("location") or {}
    energy = incident.get("energy_kwh") or {}
    features = incident.get("top_shap_features") or []
    feature_names = " ".join(str(f.get("feature", "")) for f in features)
    return (
        f"{location.get('machine_type', 'equipment')} energy anomaly investigation "
        f"actual {energy.get('actual')} expected {energy.get('expected')} "
        f"deviation {energy.get('deviation')} severity {incident.get('severity', '')} "
        f"{feature_names} operating schedule production idle HVAC motor meter maintenance "
        "energy efficiency approved procedure troubleshooting"
    )


def _rule_based_actions(incident: dict[str, Any]) -> list[dict[str, Any]]:
    """Return varied, equipment-matched checks tailored to this incident."""
    from datetime import datetime
    import hashlib

    location = incident.get("location") or {}
    energy = incident.get("energy_kwh") or {}
    features = incident.get("top_shap_features") or []
    actual = float(energy.get("actual") or 0)
    expected = float(energy.get("expected") or 0)
    deviation = float(energy.get("deviation") if energy.get("deviation") is not None else actual - expected)
    ratio = actual / expected if expected > 0 else None
    severity = str(incident.get("severity", "unknown")).lower()
    machine_type = str(location.get("machine_type", "equipment")).strip()
    machine_lower = machine_type.lower()
    machine_id = str(location.get("machine_id", "unknown"))
    timestamp = str(incident.get("timestamp", "flagged interval"))
    incident_id = str(incident.get("incident_id", ""))
    shap_text = ", ".join(str(x.get("feature", "")) for x in features[:3]) or "not available"
    shap_names = " ".join(str(x.get("feature", "")).lower() for x in features)
    priority = "High" if severity in {"high", "critical"} else "Medium"

    # Stable variation: similar incidents do not all receive the same three checks.
    seed = f"{incident_id}|{timestamp}|{actual}|{expected}|{shap_names}"
    offset = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)

    def make(text: str, reason: str, basis: str, level: str = "Medium") -> dict[str, Any]:
        return {
            "action": text,
            "priority": level,
            "reason": reason,
            "evidence_basis": basis,
            "evidence_ids": [],
            "confidence": "Investigation check only; physical cause not confirmed",
        }

    measurement = (
        f"Incident {incident_id} at {timestamp}: actual {actual:.2f} kWh, "
        f"expected {expected:.2f} kWh, deviation {deviation:+.2f} kWh"
        + (f", or {ratio:.2f} times expected." if ratio is not None else ".")
    )
    basis = f"Incident measurements and available model features: {shap_text}."

    if "furnace" in machine_lower or "kiln" in machine_lower:
        bank = [
            ("Compare furnace energy with batch throughput and batch type for the flagged shift; normalize kWh by batch output before comparing with a like-for-like run.",
             "A larger or different batch can change energy demand; compare equivalent production before concluding efficiency changed."),
            ("Review furnace temperature setpoint, warm-up/soak duration, and cycle end time around the flagged interval against a comparable normal batch.",
             "Setpoint and cycle duration can help explain an energy deviation; verify against the operator log."),
            ("Check the shift log for idle time, door-open periods, interrupted batches, and restart events around the incident timestamp.",
             "Unexpected idle or restart time is worth checking against the recorded operating sequence."),
            ("Compare the furnace trend with the top model features and investigate whether the feature values reflect a real operating change or a data-quality issue.",
             "Model features indicate what influenced the alert, not the physical cause."),
        ]
    elif "hvac" in machine_lower or "air handling" in machine_lower:
        bank = [
            ("Compare HVAC runtime with the approved schedule and occupancy for the flagged interval; check for operation outside scheduled hours.",
             "A schedule/load mismatch can increase energy use; confirm occupancy and controls history."),
            ("Review temperature setpoints and available indoor/outdoor conditions around the incident, comparing with a similar occupied period.",
             "Weather and setpoint differences affect HVAC demand; use actual site records."),
            ("Inspect HVAC alarm/control trends and recent filter or coil maintenance records for the flagged period using approved procedures.",
             "Alarms and maintenance history can guide a follow-up check without assuming a component fault."),
            ("Compare HVAC energy with operating hours and served-area demand, then check whether the baseline period had a comparable load.",
             "Like-for-like runtime and load comparisons help interpret the deviation."),
        ]
    elif "compressor" in machine_lower:
        bank = [
            ("Compare compressor discharge-pressure setpoint with actual pressure and plant air demand during the flagged interval.",
             "Pressure targets and demand should be comparable before attributing excess energy."),
            ("Review loaded versus unloaded runtime and compressor sequencing for the incident period against a normal production shift.",
             "Extended unloaded running or sequencing differences are investigation leads, not confirmed faults."),
            ("Check recent leak survey, pressure-drop, and maintenance records for the air system covering the incident timestamp.",
             "Air-system records can help prioritize checks; follow approved site procedures."),
            ("Compare compressor energy with production schedule and air demand, checking whether the baseline used a similar operating load.",
             "Different demand levels can make energy comparisons misleading."),
        ]
    elif any(token in machine_lower for token in ("motor", "pump", "fan", "conveyor", "cnc", "mill")):
        bank = [
            (f"Compare {machine_type} load or throughput and runtime during the flagged interval with a comparable production cycle.",
             "Load and runtime help distinguish a changed production demand from a possible efficiency issue."),
            (f"Review {machine_type} drive/controller alarms and speed or cycle settings around the incident timestamp.",
             "Control and alarm history may reveal an operating change; verify against site records."),
            (f"Compare maintenance and operating logs for {machine_id} with the top model features for this incident.",
             "The model features guide investigation but do not establish a physical cause."),
            (f"Check whether the production schedule, idle time, or start/stop frequency for {machine_id} differed from the baseline period.",
             "Changes in operating pattern can alter energy use even when equipment is healthy."),
        ]
    else:
        bank = [
            (f"Compare {machine_type} runtime and throughput at {timestamp} with a like-for-like operating period.",
             "A comparable load and schedule are needed to interpret the energy deviation."),
            (f"Review alarms and maintenance logs for {machine_id} around {timestamp}, using approved site/OEM procedures.",
             "Records can guide a safe inspection without assuming a fault."),
            (f"Check whether the operating schedule, idle time, or start/stop pattern for {machine_id} changed during this interval.",
             "Operating-pattern changes can affect energy use."),
            (f"Compare the incident's top model features ({shap_text}) with their normal values and confirm the source measurements.",
             "Feature attribution helps prioritize checks but is not proof of cause."),
        ]

    # Rank checks using available model features and energy context; use incident-specific
    # tie-breaking only when the evidence gives no stronger reason to prefer one check.
    ordered = _rank_asset_checks(incident, bank, offset)
    actions = []
    for text, reason in ordered[:2]:
        actions.append(make(text, f"{reason} {measurement}", basis, priority))

    # Only add meter validation as a secondary check for some incidents, not every record.
    if offset % 3 == 0:
        actions.append(make(
            f"Validate meter-to-machine mapping, units, and timestamp alignment for {machine_id} at {timestamp}.",
            f"Confirm the measurement is reliable before corrective action. {measurement}",
            basis,
            "Low",
        ))
    elif ratio is not None and ratio >= 1.2:
        actions.append(make(
            f"Compare energy per unit of production for {machine_id} with a matching batch/shift; record whether output changed during {timestamp}.",
            f"The recorded energy is {ratio:.2f} times expected; this quantifies the deviation but does not identify its cause. {measurement}",
            basis,
        ))
    else:
        actions.append(make(
            f"Check the operating log for {machine_id} at {timestamp} for a documented change in load, schedule, or cycle duration.",
            f"Use the log to confirm whether operating conditions differed from the baseline. {measurement}",
            basis,
            "Low",
        ))

    # Keep action wording unique within this incident.
    unique = []
    seen = set()
    for item in actions:
        key = re.sub(r"\W+", " ", item["action"].lower()).strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:3]


def _actions_match_equipment(incident: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    """Reject clearly wrong-asset advice; allow general checks when relevant to the incident."""
    machine_type = str((incident.get("location") or {}).get("machine_type", "equipment")).lower()
    text = " ".join(str(a.get("action", "")) for a in actions).lower()
    if not actions or not text.strip():
        return False

    # A missing asset keyword does not make a check invalid: timestamp, schedule,
    # load, production, and meter validation can apply across equipment types.
    # Reject only advice that explicitly names a different asset family.
    asset_families = {
        "hvac": ("hvac", "air handling", "air handler", "filter/coil", "occupancy"),
        "furnace": ("furnace", "kiln", "batch", "soak"),
        "kiln": ("furnace", "kiln", "batch", "soak"),
        "compressor": ("compressor", "compressed air", "discharge pressure", "loaded/unloaded"),
        "pump": ("pump", "flow rate"),
        "fan": ("fan", "airflow"),
        "motor": ("motor", "drive/controller"),
    }
    current_family = next((family for family in asset_families if family in machine_type), None)
    if current_family:
        other_terms = {
            term
            for family, terms in asset_families.items()
            if family != current_family
            for term in terms
        }
        # HVAC occupancy is not enough on its own to prove a mismatch; treat it as
        # specific only when paired with HVAC schedule/filter/coil wording.
        explicit_other = [
            term for term in other_terms
            if term in text and term not in {"occupancy"}
        ]
        if explicit_other:
            current_terms = asset_families[current_family]
            if not any(term in text for term in current_terms):
                return False
    return True


def _filter_evidence_for_equipment(
    incident: dict[str, Any], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prefer equipment-matched passages; retain only broadly applicable guidance otherwise."""
    machine = str((incident.get("location") or {}).get("machine_type", "equipment")).lower()
    groups = {
        "furnace": (("furnace", "kiln"), ("hvac", "air handling", "compressor", "motor-driven")),
        "kiln": (("furnace", "kiln"), ("hvac", "air handling", "compressor", "motor-driven")),
        "hvac": (("hvac", "air handling", "filter", "coil"), ("furnace", "kiln", "compressor", "motor-driven")),
        "air handling": (("hvac", "air handling", "filter", "coil"), ("furnace", "kiln", "compressor", "motor-driven")),
        "compressor": (("compressor", "compressed air", "air system"), ("hvac", "air handling", "furnace", "kiln")),
        "motor": (("motor", "drive", "motor-driven"), ("hvac", "air handling", "furnace", "kiln", "compressor")),
        "pump": (("pump", "motor-driven", "drive"), ("hvac", "air handling", "furnace", "kiln", "compressor")),
        "fan": (("fan", "motor-driven", "drive"), ("hvac", "air handling", "furnace", "kiln", "compressor")),
    }
    applicable, exclude = (), ()
    for key, (good, bad) in groups.items():
        if key in machine:
            applicable, exclude = good, bad
            break
    if not applicable:
        return retrieved

    matched, general = [], []
    for item in retrieved:
        section = str(item.get("section", "")).lower()
        excerpt = str(item.get("excerpt", "")).lower()
        text = section + " " + excerpt
        # Keep broadly applicable incident guidance, but exclude explicit guidance for another asset.
        if any(term in section for term in exclude):
            continue
        if any(term in section for term in applicable):
            matched.append(item)
        elif any(term in section for term in ("unexpectedly high energy", "energy increase", "general", "data quality")):
            general.append(item)
        elif not any(term in text for term in exclude):
            general.append(item)
    selected = (matched + general)[:4]
    # If all candidates were excluded, retain original retrieval rather than returning no context.
    return selected or retrieved[:2]


def _rank_asset_checks(
    incident: dict[str, Any],
    bank: list[tuple[str, str]],
    offset: int,
) -> list[tuple[str, str]]:
    """Rank existing safe checks using feature names and incident context; don't infer a fault."""
    features = incident.get("top_shap_features") or []
    names = " ".join(str(item.get("feature", "")).lower() for item in features)
    energy = incident.get("energy_kwh") or {}
    actual = float(energy.get("actual") or 0)
    expected = float(energy.get("expected") or 0)
    ratio = actual / expected if expected > 0 else 0.0

    def score(pair: tuple[str, str]) -> tuple[int, int]:
        action = pair[0].lower()
        points = 0
        if "production_output" in names or "production" in names or "throughput" in names:
            if any(word in action for word in ("throughput", "batch", "production", "output", "per unit")):
                points += 4
        if "operating_status" in names or "runtime" in names or "running" in names:
            if any(word in action for word in ("runtime", "operating hours", "idle", "start/stop", "schedule", "shift log")):
                points += 3
        if any(word in names for word in ("temperature", "setpoint", "thermal")):
            if any(word in action for word in ("temperature", "setpoint", "warm-up", "soak", "cycle")):
                points += 4
        if any(word in names for word in ("pressure", "air_demand", "compressor")):
            if any(word in action for word in ("pressure", "air demand", "loaded", "unloaded", "compressor")):
                points += 4
        if ratio >= 1.4 and any(word in action for word in ("per unit", "throughput", "load", "demand")):
            points += 1
        # Stable incident-specific tie-breaker; not used as evidence of cause.
        tie = int(hashlib.sha256(f"{incident.get('incident_id')}|{offset}|{action}".encode()).hexdigest()[:8], 16)
        return points, -tie

    return sorted(bank, key=score, reverse=True)

def _severity_priority(severity: str) -> str:
    return {"critical": "Immediate review", "high": "Prompt review", "medium": "Review", "low": "Routine review"}.get(severity.strip().lower(), "Review")


def _llm_actions(incident: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Generate recommendations with Nugen aligned inference by default, or local Ollama."""
    import urllib.error
    import urllib.request
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.getenv("EMISSIONEX_LLM_PROVIDER", "nugen").strip().lower()
    if provider == "nugen":
        api_key = os.getenv("NUGEN_API_KEY", "").strip()
        endpoint = os.getenv("NUGEN_MODEL_ENDPOINT", "").strip()
        model = os.getenv("NUGEN_MODEL_ID", "").strip()
        if not model and endpoint:
            last_part = endpoint.rstrip("/").rsplit("/", 1)[-1]
            if last_part.startswith("model_"):
                model = last_part
        if not api_key or not endpoint or not model:
            print("WARNING: Nugen selected but NUGEN_API_KEY, NUGEN_MODEL_ENDPOINT, or NUGEN_MODEL_ID is missing; using conservative fallback.")
            return None
        base_url = endpoint
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    else:
        print(f"WARNING: Unknown EMISSIONEX_LLM_PROVIDER={provider!r}; using conservative fallback.")
        return None
    allowed_evidence = []
    for idx, item in enumerate(evidence[:3], start=1):
        allowed_evidence.append({
            "evidence_id": f"E{idx}",
            "document": str(item.get("document", ""))[:100],
            "section": str(item.get("section", ""))[:100],
            "excerpt": str(item.get("excerpt", ""))[:350],
        })

    location = incident.get("location") or {}
    energy = incident.get("energy_kwh") or {}
    features = incident.get("top_shap_features") or []
    compact_incident = {
        "incident_id": incident.get("incident_id"),
        "machine_type": location.get("machine_type", "equipment"),
        "machine_id": location.get("machine_id", "unknown"),
        "severity": incident.get("severity", "unknown"),
        "actual_kwh": energy.get("actual"),
        "expected_kwh": energy.get("expected"),
        "deviation_kwh": energy.get("deviation"),
        "top_features": [str(f.get("feature", "")) for f in features[:3]],
    }

    schema = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "string", "enum": ["Low", "Medium", "High"]},
                        "reason": {"type": "string"},
                        "evidence_basis": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string", "enum": [e["evidence_id"] for e in allowed_evidence]} if allowed_evidence else {"type": "string"}},
                        "confidence": {"type": "string"},
                    },
                    "required": ["action", "priority", "reason", "evidence_basis", "evidence_ids", "confidence"],
                },
            }
        },
        "required": ["actions"],
    }
    prompt = (
        "Investigate this factory energy anomaly. Return exactly 2 concise practical checks, each under 25 words, specific to this equipment and incident. "
        "Do not repeat or paraphrase the same check. Avoid generic boilerplate and do not mention equipment types absent from the incident. "
"Use different checks suited to the supplied machine_type, measurements, timestamp, and top_features. Output only the required JSON object. Use only evidence IDs supplied. "
        "If a passage supports an action, cite its ID and name its section in evidence_basis. "
        "If none supports it, use empty evidence_ids and say 'Incident data; verify on site' in evidence_basis. "
        "SHAP is prediction attribution, not proof of physical cause. Never invent manual details, "
        "recommend bypassing safety systems, or instruct work on energized equipment.\n"
        "INCIDENT=" + json.dumps(compact_incident, ensure_ascii=False) +
        "\nEVIDENCE=" + json.dumps(allowed_evidence, ensure_ascii=False)
    )
    if provider == "nugen":
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious industrial energy investigation assistant. "
                        "Return only valid JSON following the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        url = base_url
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "EmissioNex/1.0",
        }
        timeout = int(os.getenv("NUGEN_TIMEOUT_SECONDS", "120"))
    else:
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cautious industrial energy investigation assistant. Follow the JSON schema exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0, "num_predict": 500, "num_ctx": 2048},
        }
        url = f"{base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        print("Nugen request URL:", request.full_url, flush=True)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if provider == "ollama":
            content = (raw.get("message") or {}).get("content", "")
        else:
            choices = raw.get("choices") or []
            content = ((choices[0].get("message") or {}).get("content", "") if choices else "")
        if not content:
            print(f"WARNING: {provider} response contained no message content; using conservative fallback.")
            return None
        content = content.strip()

        # Normalize common wrappers returned by hosted chat models: fenced JSON,
        # a leading `json` label, or a JSON-encoded string containing JSON text.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        content = re.sub(r"^json\s*", "", content, flags=re.IGNORECASE).strip()
        for _ in range(2):
            if not content.startswith('"'):
                break
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                break
            if not isinstance(decoded, str):
                break
            content = decoded.strip()
            content = re.sub(r"^json\s*", "", content, flags=re.IGNORECASE).strip()

        # If JSON text was returned with literal escaped newlines/quotes, decode
        # the wrapper only when that produces a plausible JSON object/array.
        if content.startswith(("{", "[")):
            candidates = [content]
            if r"\n" in content or r'\"' in content:
                try:
                    decoded = json.loads('"' + content.replace('\\', '\\\\').replace('"', '\\"') + '"')
                    if isinstance(decoded, str):
                        candidates.insert(0, decoded)
                except json.JSONDecodeError:
                    pass
        else:
            candidates = [content]
            first_obj = content.find("{")
            first_arr = content.find("[")
            starts = [i for i in (first_obj, first_arr) if i >= 0]
            if starts:
                candidates.insert(0, content[min(starts):])

        parsed = None
        parse_error = None
        for candidate in candidates:
            # Trim trailing explanatory text by finding a balanced JSON value.
            candidate = candidate.strip()
            if candidate.startswith(("{", "[")):
                opener = candidate[0]
                closer = "}" if opener == "{" else "]"
                depth = 0
                in_string = escaped = False
                for i, ch in enumerate(candidate):
                    if in_string:
                        if escaped:
                            escaped = False
                        elif ch == "\\":
                            escaped = True
                        elif ch == '"':
                            in_string = False
                    elif ch == '"':
                        in_string = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = candidate[:i + 1]
                            break
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc
        if parsed is None:
            raise json.JSONDecodeError(str(parse_error or "No JSON found"), content, 0)

        if isinstance(parsed, list):
            actions = parsed
        elif isinstance(parsed, dict):
            candidate = parsed.get("actions")
            if candidate is None:
                candidate = parsed.get("action")
            # Nugen may return checks[] plus shared/per-check evidence fields.
            # Normalize that schema into the action objects used below.
            if candidate is None and isinstance(parsed.get("checks"), list):
                checks = parsed["checks"]
                ids_value = parsed.get("evidence_ids", [])
                basis_value = parsed.get("evidence_basis", "")
                actions = []
                for idx, check in enumerate(checks):
                    if isinstance(check, dict):
                        action_obj = dict(check)
                    elif isinstance(check, str):
                        action_obj = {"action": check}
                    else:
                        continue
                    if "evidence_ids" not in action_obj:
                        action_obj["evidence_ids"] = (
                            ids_value[idx] if isinstance(ids_value, list)
                            and ids_value and isinstance(ids_value[0], list)
                            and idx < len(ids_value) else ids_value
                        )
                    if "evidence_basis" not in action_obj:
                        action_obj["evidence_basis"] = (
                            basis_value[idx] if isinstance(basis_value, list)
                            and idx < len(basis_value) else basis_value
                        )
                    actions.append(action_obj)
            elif isinstance(candidate, list):
                actions = candidate
            elif isinstance(candidate, dict):
                nested = candidate.get("action")
                if isinstance(nested, dict):
                    actions = [nested]
                elif isinstance(nested, str):
                    actions = [candidate]
                else:
                    actions = [candidate]
            elif isinstance(candidate, str):
                actions = [parsed]
            else:
                actions = []
        else:
            actions = []

        if not actions:
            print(f"WARNING: {provider} response had no action list; using conservative fallback.")
            return None

        valid_ids = {item["evidence_id"] for item in allowed_evidence}
        evidence_by_id = {item["evidence_id"]: item for item in allowed_evidence}
        cleaned: list[dict[str, Any]] = []
        for action in actions[:2]:
            if not isinstance(action, dict) or not action.get("action"):
                continue
            ids = [str(x) for x in action.get("evidence_ids", []) if str(x) in valid_ids]
            priority = str(action.get("priority", "Medium")).title()
            if priority not in {"Low", "Medium", "High"}:
                priority = "Medium"
            action_text = str(action["action"])[:500]
            reason = str(action.get("reason", ""))[:500]
            evidence_basis = str(action.get("evidence_basis", ""))[:500]

            # If the small model omits citations, attach one only when the action/reason
            # clearly overlaps the retrieved passage. Never fabricate an evidence ID.
            if not ids and allowed_evidence:
                words = set(re.findall(r"[a-z]{4,}", f"{action_text} {reason}".lower()))
                stop = {"check", "review", "verify", "ensure", "should", "could", "from", "with", "that", "this", "when", "into", "against", "based", "action", "energy"}
                words -= stop
                scored = []
                for item in allowed_evidence:
                    evidence_words = set(re.findall(
                        r"[a-z]{4,}",
                        f"{item.get('section', '')} {item.get('excerpt', '')}".lower()
                    )) - stop
                    overlap = len(words & evidence_words)
                    if overlap:
                        scored.append((overlap, item))
                if scored:
                    scored.sort(key=lambda pair: pair[0], reverse=True)
                    best_score, best_item = scored[0]
                    if best_score >= 2:
                        ids = [best_item["evidence_id"]]
                        evidence_basis = evidence_basis or f"Retrieved section: {best_item.get('section', '')}"

            cleaned.append({
                "action": action_text,
                "priority": priority,
                "reason": reason,
                "evidence_basis": evidence_basis or "Incident data; verify on site.",
                "evidence_ids": ids,
                "confidence": str(action.get("confidence", "Needs human verification"))[:200],
            })
        # Remove exact and near-duplicate actions before returning them.
        deduped: list[dict[str, Any]] = []
        seen_actions: set[str] = set()
        for item in cleaned:
            normalized = re.sub(r"\W+", " ", item["action"].lower()).strip()
            # Compare meaningful words so small wording changes don't repeat the same check.
            words = set(re.findall(r"[a-z0-9]{3,}", normalized))
            duplicate = False
            for prior in deduped:
                prior_norm = re.sub(r"\W+", " ", prior["action"].lower()).strip()
                prior_words = set(re.findall(r"[a-z0-9]{3,}", prior_norm))
                if normalized == prior_norm or (words and prior_words and len(words & prior_words) / len(words | prior_words) >= 0.72):
                    duplicate = True
                    break
            if not duplicate and normalized not in seen_actions:
                seen_actions.add(normalized)
                deduped.append(item)
        return deduped or None
    except urllib.error.HTTPError as e:
        print("Nugen status:", e.code)
        print("Nugen request URL:", e.geturl(), flush=True)
        print("Nugen error body:", e.read().decode("utf-8", errors="replace"))
        # Do not abort the whole run on an endpoint/auth/server error; use the
        # conservative rule-based checks and record fallback mode instead.
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"WARNING: model inference failed; using conservative fallback ({exc}).")
        return None
    except Exception as exc:
        print(f"WARNING: model inference failed; using conservative fallback ({exc}).")
        return None


def generate_recommendations(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
    kb_dir: Path = Path("knowledge_base/reference_guides"),
    retrieval_backend: str | None = None,
    use_llm: bool | None = None,
) -> list[dict[str, Any]]:
    with Path(input_path).open("r", encoding="utf-8") as f:
        incidents = json.load(f)
    if not isinstance(incidents, list):
        raise ValueError("Expected forensic_incidents.json to contain a JSON list.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_backend = retrieval_backend or os.getenv("EMISSIONEX_RAG_BACKEND", "auto")
    rag = LocalRAG(kb_dir, db_dir=output_dir / "chroma_db", backend=retrieval_backend)
    if use_llm is None:
        # Nugen is the default provider. Set EMISSIONEX_LLM_PROVIDER=ollama only for local testing.
        # EMISSIONEX_USE_LLM=0 forces rule-based fallback and is not appropriate if all inference must use Nugen.
        use_llm = os.getenv("EMISSIONEX_USE_LLM", "1").strip().lower() not in {"0", "false", "no", "off"}

    # Safe quick-test mode: process 3 incidents by default.
    # Set EMISSIONEX_MAX_INCIDENTS=0 to process the complete dataset.
    max_incidents = int(os.getenv("EMISSIONEX_MAX_INCIDENTS", "3"))
    if max_incidents > 0:
        incidents = incidents[:max_incidents]
    print(f"Incidents queued: {len(incidents)}")
    print(f"LLM inference enabled: {bool(use_llm)}")
    print("LLM provider:", os.getenv("EMISSIONEX_LLM_PROVIDER", "nugen").strip().lower())
    print("Starting recommendation generation...")

    results: list[dict[str, Any]] = []
    for incident_index, incident in enumerate(incidents, start=1):
        print(f"[{incident_index}/{len(incidents)}] Processing {incident.get('incident_id', 'unknown incident')}...", flush=True)
        location = incident.get("location") or {}
        energy = incident.get("energy_kwh") or {}
        retrieved = rag.search(_build_query(incident), top_k=8, min_relevance=0.10)
        retrieved = _filter_evidence_for_equipment(incident, retrieved)
        evidence = []
        for idx, item in enumerate(retrieved, start=1):
            evidence.append({
                "evidence_id": f"E{idx}",
                "document": item["document"],
                "section": item["section"],
                "source_path": item["source"],
                "page": item.get("page"),
                "relevance_score": item["relevance_score"],
                "retrieval_backend": item.get("retrieval_backend", rag.backend),
                "excerpt": item["excerpt"],
            })

        actions = _llm_actions(incident, evidence) if use_llm else None
        if actions and not _actions_match_equipment(incident, actions):
            print("WARNING: Generic or equipment-mismatched AI actions rejected; using tailored checks.")
            actions = None
        generation_mode = (os.getenv("EMISSIONEX_LLM_PROVIDER", "nugen").strip().lower() + "_llm_grounded") if actions else "rule_based_fallback"
        if not actions:
            actions = _rule_based_actions(incident)
        severity = str(incident.get("severity", "unknown"))
        evidence_status = (
            "Retrieved evidence available; confirm against approved site/OEM documents."
            if evidence else
            "No sufficiently relevant reference passage retrieved; actions are generic investigation checks only."
        )
        results.append({
            "incident_id": incident.get("incident_id"),
            "timestamp": incident.get("timestamp"),
            "location": location,
            "severity": severity,
            "review_priority": _severity_priority(severity),
            "energy_kwh": energy,
            "excess_co2e_kg": incident.get("excess_co2e_kg"),
            "baseline_model_id": incident.get("baseline_model_id"),
            "top_shap_features": incident.get("top_shap_features", []),
            "retrieval_backend": rag.backend,
            "generation_mode": generation_mode,
            "recommended_actions": actions,
            "retrieved_evidence": evidence,
            "evidence_status": evidence_status,
            "limitations": (
                "Advisory prototype. SHAP contributions and energy deviations do not prove a physical cause. "
                "Only evidence IDs tied to retrieved passages should be treated as document citations. "
                "A qualified person must verify conditions and follow approved safety procedures."
            ),
        })

    with (output_dir / "recommendations.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for record in results:
        for rank, action in enumerate(record["recommended_actions"], start=1):
            flat_rows.append({
                "incident_id": record["incident_id"],
                "timestamp": record["timestamp"],
                "machine_id": (record.get("location") or {}).get("machine_id"),
                "severity": record["severity"],
                "review_priority": record["review_priority"],
                "action_rank": rank,
                "action": action["action"],
                "priority": action["priority"],
                "reason": action["reason"],
                "evidence_basis": action["evidence_basis"],
                "evidence_ids": "; ".join(action.get("evidence_ids", [])),
                "source_documents": "; ".join(
                    f'{s["document"]} — {s["section"]} (p.{s["page"]})' if s.get("page") else f'{s["document"]} — {s["section"]}'
                    for s in record["retrieved_evidence"]
                ),
                "retrieval_backend": record["retrieval_backend"],
                "generation_mode": record["generation_mode"],
            })
    pd.DataFrame(flat_rows).to_csv(output_dir / "recommendations.csv", index=False)

    with (output_dir / "recommendation_report.txt").open("w", encoding="utf-8") as f:
        f.write("EMISSIONEX — EVIDENCE-GROUNDED RECOMMENDATION REPORT\n")
        f.write("=" * 62 + "\n")
        f.write(f"Incidents processed: {len(results)}\n")
        f.write(f"Knowledge-base chunks indexed: {len(rag.documents)}\n")
        f.write(f"Retrieval backend: {rag.backend}\n")
        f.write(f"LLM inference requested: {bool(use_llm)}\n")
        f.write("LLM provider: " + os.getenv("EMISSIONEX_LLM_PROVIDER", "nugen").strip().lower() + "\n")
        f.write("The bundled guide is demonstration content, not an OEM manual or regulatory standard.\n\n")
        for record in results:
            loc = record.get("location") or {}
            energy = record.get("energy_kwh") or {}
            f.write(f"Incident: {record.get('incident_id')}\n")
            f.write(f"Time: {record.get('timestamp')}\n")
            f.write(f"Machine: {loc.get('machine_id', 'Unknown')} ({loc.get('machine_type', 'Unknown')})\n")
            f.write(f"Severity: {record.get('severity')} | Actual: {energy.get('actual')} kWh | Expected: {energy.get('expected')} kWh\n")
            f.write(f"Excess CO2e: {record.get('excess_co2e_kg')} kg\n")
            f.write(f"Mode: {record['retrieval_backend']} / {record['generation_mode']}\n")
            f.write("Recommended actions:\n")
            for action in record["recommended_actions"]:
                f.write(f"  - [{action['priority']}] {action['action']}\n")
                if action.get("evidence_ids"):
                    f.write(f"    Evidence IDs: {', '.join(action['evidence_ids'])}\n")
            f.write("Retrieved evidence:\n")
            for source in record["retrieved_evidence"]:
                page = f", page {source['page']}" if source.get("page") else ""
                f.write(f"  - [{source['evidence_id']}] {source['document']} / {source['section']}{page}; score {source['relevance_score']}\n")
            f.write("Caution: recommendations are advisory; verify causes and follow approved procedures.\n")
            f.write("-" * 62 + "\n")

    print("Recommendation generation complete.")
    print("Incidents processed:", len(results))
    print("Knowledge-base chunks:", len(rag.documents))
    print("Retrieval backend:", rag.backend)
    print("Generation mode(s):", sorted({r["generation_mode"] for r in results}))
    print("JSON:", output_dir / "recommendations.json")
    print("CSV:", output_dir / "recommendations.csv")
    print("Report:", output_dir / "recommendation_report.txt")
    return results


if __name__ == "__main__":
    generate_recommendations()

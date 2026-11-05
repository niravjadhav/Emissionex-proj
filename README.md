# Emissionex-proj
# EmissioNex — Anomaly Detection, Explainability, Localization & Evidence-Grounded Recommendations

EmissioNex investigates unusual energy/CO₂e increases, localizes them in the facility hierarchy, explains which model features influenced expected energy, and produces advisory corrective-action recommendations supported by retrieved technical references.

A statistical anomaly is an investigation lead, not proof of a physical fault. SHAP explains a model prediction; it does not establish a physical cause.

## Quick start

Use Python 3.11 or 3.12 recommended (3.10–3.13 may work depending on dependency availability). Open a terminal in the folder containing `main.py` and `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the existing analytics and explainability pipeline if needed:

```bash
python main.py
python explainability.py
```

Generate recommendations from the existing forensic JSON:

```bash
python recommendation_engine.py
```

## Recommendation architecture

1. **Incident context:** reads `outputs/forensic_incidents.json` (location, severity, actual/expected energy, CO₂e deviation, and SHAP features).
2. **Document ingestion and chunking:** reads `.md`, `.txt`, and `.pdf` from `knowledge_base/reference_guides/`; chunks text while retaining source, heading, and PDF page metadata.
3. **Semantic retrieval (primary):** `sentence-transformers/all-MiniLM-L6-v2` embeddings are stored in a local ChromaDB collection under `outputs/chroma_db/`.
4. **Transparent fallback:** if ChromaDB, Sentence Transformers, or the embedding model cannot be loaded, `EMISSIONEX_RAG_BACKEND=auto` falls back to TF-IDF and prints a warning. Each output records its actual retrieval backend. To require semantic mode, use `EMISSIONEX_RAG_BACKEND=semantic`; it raises an actionable error rather than silently falling back.
5. **LLM recommendation generation (optional):** the current project run uses the Nugen inference endpoint. The engine sends incident context and retrieved evidence and accepts structured, evidence-grounded recommendations. Each incident records its generation mode. If inference fails or the response is unusable, the engine uses conservative rule-based checks and records `generation_mode=rule_based_fallback`.
6. **Outputs:** JSON, CSV and a readable report with generation mode, retrieval backend, source document, section/page when available, evidence excerpts, and limitations. The dashboard reads `outputs/recommendations.json` and displays the recommendation source and evidence alongside each incident.

### Configure Nugen inference

Configure the Nugen API credentials and model settings in your local `.env` file, following the variable names expected by `recommendation_engine.py`. Never commit or share `.env`, and do not expose API keys in screenshots or logs. Nugen inference requires an internet connection and may be subject to service latency or connection failures. If a request fails or its response cannot be used, the engine falls back to conservative rule-based checks and records that mode; it should not label those checks as LLM-generated.

The current full run processed 443 incidents: 377 used `nugen_llm_grounded` (85.1%) and 66 used `rule_based_fallback` (14.9%). This is a run result, not a guaranteed rate for future runs.

### Semantic retrieval settings

The first semantic run may download the embedding model and can take longer. Internet access is needed the first time unless the model is already cached. Set the optional values in `.env`:

```text
EMISSIONEX_RAG_BACKEND=auto
EMISSIONEX_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

- `auto`: use ChromaDB + embeddings when available, otherwise clearly report TF-IDF fallback.
- `semantic`: require semantic retrieval and fail with an error if unavailable.
- `tfidf`: force lightweight lexical retrieval for quick/offline tests.

### Add knowledge documents

Place approved manufacturer manuals, facility SOPs, and relevant energy-management guidance under `knowledge_base/reference_guides/`. PDFs are parsed with `pypdf`. Keep source documents licensed and approved for project use. The current `demo_energy_investigation_guide.md` is synthetic demonstration content, **not** an OEM manual, regulatory standard, or independently verified technical source. Replace/supplement it before claiming equipment-specific advice.

## Recommendation outputs

| Output | Purpose |
|---|---|
| `outputs/recommendations.json` | Structured recommendations per incident, retrieved evidence, source metadata, retrieval/generation modes, and limitations. |
| `outputs/recommendations.csv` | Flat action list for analysis and dashboard consumption. |
| `outputs/recommendation_report.txt` | Human-readable report. |
| `outputs/chroma_db/` | Local semantic vector index, generated when semantic mode is available. |

The Streamlit dashboard reads the structured JSON output rather than parsing the text report. The Incident Workspace displays the recommendation source, recommended steps, and evidence for the selected incident. The current UI has been checked with both a Nugen-grounded incident and a rule-based fallback incident.

## Run the dashboard

Install the dependencies from `requirements.txt`, then launch the Streamlit app from the project folder:

```powershell
python -m streamlit run app.py
```

If PowerShell reports that `streamlit` is not recognized, use the `python -m streamlit` form above (and confirm the project environment has its dependencies installed).

## Evaluation and tests

```bash
python rag_evaluation.py
python -m pytest -q tests/test_recommendation_engine.py
```

`rag_evaluation.py` is a small retrieval smoke test, not a formal accuracy benchmark. Evaluate the system against manually reviewed query/document pairs before reporting quantitative RAG quality. The recommendation-engine tests force TF-IDF for repeatability; they do not prove that the semantic backend or external LLM works in every environment.

## Main project pipeline

1. `preprocessing.py` validates/coerces telemetry and preserves source row identifiers.
2. `baseline.py` predicts expected kWh using XGBoost models.
3. `anomaly_detection.py` identifies unusual residuals.
4. `localization.py` rolls incidents up through facility hierarchy and time windows.
5. `evaluation.py` compares detections with known labels for evaluation.
6. `explainability.py` applies SHAP to saved models and exports feature explanations.
7. `recommendation_engine.py` retrieves reference passages and generates advisory actions.

## Important limitations

- The supplied evaluation report notes that baseline R² and fault recall do not yet meet all original targets. Do not present the detector as having met those targets.
- The supplied data contains limited injected fault episodes and is suitable for a prototype, not a validated industrial diagnosis system.
- The current hierarchy localizes to machine level; it cannot identify an internal component without additional telemetry.
- Recommendation quality depends on the quality and relevance of indexed reference documents. The bundled demonstration guide is not an official technical source.
- LLM output can be wrong. Evidence references, source excerpts, and human review are required. The system must not claim a physical cause based on SHAP alone.
- Do not send confidential facility data to a cloud LLM unless the organization has approved that use.

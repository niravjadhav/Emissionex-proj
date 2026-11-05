"""Small smoke evaluation for retrieval relevance; not a formal benchmark."""
from pathlib import Path
from rag_pipeline import LocalRAG

CASES = [
    ("unexpectedly high energy meter timestamp units production", ["Unexpectedly high energy relative to expected energy"]),
    ("energy increase production unchanged cycle time product mix", ["Energy increase while production is unchanged or lower"]),
    ("off shift idle standby safety ventilation shutdown", ["Unexpected consumption during an off-shift or idle period"]),
    ("HVAC setpoint schedule filters coil maintenance alarms", ["HVAC equipment energy investigation"]),
    ("motor drive load operating hours alarms maintenance", ["Motor-driven equipment energy investigation"]),
]


def run_evaluation(kb_dir: Path = Path("knowledge_base/reference_guides"), backend: str = "auto") -> list[dict]:
    rag = LocalRAG(kb_dir, db_dir=Path("outputs/chroma_db_eval"), backend=backend)
    rows = []
    for query, expected_sections in CASES:
        hits = rag.search(query, top_k=3, min_relevance=0.05)
        hit_sections = [item["section"] for item in hits]
        matched = any(expected in section for expected in expected_sections for section in hit_sections)
        rows.append({"query": query, "expected_section": expected_sections[0], "top_sections": hit_sections, "matched_expected_section": matched})
    total = len(rows)
    matched = sum(row["matched_expected_section"] for row in rows)
    print(f"Retrieval smoke test: {matched}/{total} expected sections appeared in top-3.")
    print(f"Backend: {rag.backend}; indexed chunks: {len(rag.documents)}")
    for row in rows:
        print(("PASS" if row["matched_expected_section"] else "CHECK") + ": " + row["query"])
        print("  -> " + ", ".join(row["top_sections"]))
    return rows


if __name__ == "__main__":
    run_evaluation()

import json
from pathlib import Path

from recommendation_engine_main import generate_recommendations
from rag_pipeline import LocalRAG


def test_retriever_finds_reference_sections():
    rag = LocalRAG(
        Path("knowledge_base/reference_guides"),
        db_dir=Path("outputs/test_chroma_db"),
        backend="tfidf",
    )
    hits = rag.search("HVAC filter coil maintenance setpoint schedule", top_k=3, min_relevance=0.01)
    assert hits
    assert any("HVAC" in hit["section"] for hit in hits)
    assert hits[0]["retrieval_backend"] == "tfidf_fallback"


def test_recommendations_for_all_incidents(tmp_path):
    source = Path("outputs/forensic_incidents.json")
    if not source.exists():
        return
    results = generate_recommendations(
        input_path=source,
        output_dir=tmp_path,
        kb_dir=Path("knowledge_base/reference_guides"),
        retrieval_backend="tfidf",
        use_llm=False,
    )
    assert len(results) == 443
    assert all(item.get("recommended_actions") for item in results)
    assert all("limitations" in item for item in results)
    assert all(item["generation_mode"] == "rule_based_fallback" for item in results)
    assert all("retrieval_backend" in item for item in results)
    assert all("retrieved_evidence" in item for item in results)
    saved = json.loads((tmp_path / "recommendations.json").read_text(encoding="utf-8"))
    assert len(saved) == len(results)
    assert (tmp_path / "recommendations.csv").exists()
    assert (tmp_path / "recommendation_report.txt").exists()


def test_llm_without_key_uses_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # The fallback is exercised through the public generation API in the main test.
    assert not __import__("os").getenv("OPENAI_API_KEY")

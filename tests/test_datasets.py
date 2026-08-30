from adr.datasets.loader import load_queries


def test_load_drb_english_slice():
    rows = load_queries("deep_research_bench", language="en", limit=3)
    assert len(rows) == 3
    assert all(r.language == "en" for r in rows)
    assert all(r.text for r in rows)
    assert rows[0].id == "51"


def test_load_gym_by_id():
    rows = load_queries("deep_research_gym", query_ids=["923549"])
    assert len(rows) == 1
    assert "chip shortage" in rows[0].text

"""The Gym backend must match the hosted API contract at clueweb22.us/openapi.json."""

from __future__ import annotations

import subprocess
import sys

import pytest

from adr.tools.search import GymSearch, MockSearch, _gym_hits, build_search


def test_import_order_does_not_break():
    """tools.search imports core.state, so core must not import tools at runtime."""
    for first in ("adr.tools.search", "adr.core"):
        subprocess.run(
            [sys.executable, "-c", f"import {first}; import adr.core, adr.tools.search"],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "corpus,expected_path,cw22_a",
    [
        ("fineweb", "/fineweb/search", False),
        ("clueweb22", "/search", False),
        ("clueweb22-b", "/search", False),
        ("clueweb22-a", "/search", True),
    ],
)
def test_corpus_selects_the_route(corpus, expected_path, cw22_a):
    backend = GymSearch(corpus=corpus)
    assert backend.search_url == f"https://clueweb22.us{expected_path}"
    assert backend.cw22_a is cw22_a


def test_unknown_corpus_rejected_early():
    with pytest.raises(ValueError, match="Unknown Gym corpus"):
        GymSearch(corpus="bogus")


def test_api_key_travels_as_header_not_query_param():
    backend = GymSearch(api_key="secret", corpus="fineweb")
    headers = backend._headers()
    assert headers["x-api-key"] == "secret"
    assert "Authorization" not in headers


def test_fetch_is_disabled_by_default():
    """The hosted API exposes no archival fetch route."""
    assert GymSearch().fetch_url == ""


async def test_fetch_returns_empty_rather_than_raising():
    assert await GymSearch().fetch("https://example.com/a") == ""


def test_hits_map_clueweb_field_names():
    payload = {
        "results": [
            {
                "URL": "https://example.com/a",
                "ClueWeb22-ID": "clueweb22-en0000-00-00000",
                "Clean-Text": "body text about chips",
                "title": "Chips",
                "distance": 0.25,
            },
            {"url": "https://example.com/b", "text": "second", "distance": 0.75},
            {"text": "dropped because it has no url"},
        ]
    }
    hits = _gym_hits(payload, k=5)
    assert [h.url for h in hits] == ["https://example.com/a", "https://example.com/b"]
    assert hits[0].doc_id == "clueweb22-en0000-00-00000"
    assert hits[0].text == "body text about chips"
    # Nearer documents must rank above farther ones.
    assert hits[0].score > hits[1].score


def test_hits_respect_k():
    payload = {"results": [{"url": f"https://example.com/{i}"} for i in range(10)]}
    assert len(_gym_hits(payload, k=3)) == 3


def test_build_search_dispatch():
    assert isinstance(build_search({"backend": "mock"}), MockSearch)
    assert isinstance(build_search({"backend": "gym"}), GymSearch)
    with pytest.raises(ValueError, match="Unknown search backend"):
        build_search({"backend": "nope"})

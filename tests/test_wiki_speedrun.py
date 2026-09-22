"""Tests for the Wikipedia speedrun helpers."""

from io import BytesIO
from unittest.mock import patch

from src.wiki_speedrun import get_wikipedia_links


def test_get_wikipedia_links_returns_existing_articles() -> None:
    """Exclude missing pages and links outside the article namespace."""
    response = BytesIO(
        b'{"parse":{"links":['
        b'{"ns":0,"title":"Python","exists":true},'
        b'{"ns":14,"title":"Category:Programming","exists":true},'
        b'{"ns":0,"title":"Missing page"}'
        b"]}}"
    )

    with patch("src.wiki_speedrun.urlopen", return_value=response):
        links = get_wikipedia_links("Computer science")

    assert links == ["Python"]

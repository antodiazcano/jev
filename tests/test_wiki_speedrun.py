"""Tests for the Wikipedia speedrun helpers."""

import json
from email.message import Message
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from src.classifiers import LinkClassifier
from src.wiki_speedrun import WikipediaSpeedrun


class FirstLinkClassifier(LinkClassifier):
    """Always select the first available link."""

    def _select_from_batch(self, links: list[str]) -> str:
        return links[0]


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
        links = WikipediaSpeedrun._get_wikipedia_links("Computer science")

    assert links == ["Python"]


def test_get_wikipedia_links_retries_rate_limit() -> None:
    """Wait for Retry-After and retry a rate-limited Wikipedia request."""
    headers = Message()
    headers["Retry-After"] = "2"
    rate_limit = HTTPError("url", 429, "Too Many Requests", headers, None)
    response = BytesIO(b'{"parse":{"links":[{"ns":0,"title":"Python","exists":true}]}}')

    with (
        patch("src.wiki_speedrun.urlopen", side_effect=[rate_limit, response]),
        patch("src.wiki_speedrun.time.sleep") as sleep,
    ):
        links = WikipediaSpeedrun._get_wikipedia_links("Computer science")

    assert links == ["Python"]
    sleep.assert_called_once_with(2)


def test_predict_reaches_target(capsys, monkeypatch, tmp_path) -> None:
    """Report the route when the classifier reaches the target."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    speedrun = WikipediaSpeedrun(
        "Start", "Target", FirstLinkClassifier("Start", "Target")
    )

    with (
        patch.object(
            WikipediaSpeedrun,
            "_get_wikipedia_links",
            side_effect=[["Middle"], ["Target"]],
        ),
        patch("src.wiki_speedrun.time.time", side_effect=[10.0, 12.5]),
    ):
        speedrun.predict(max_attempts=2)

    assert capsys.readouterr().out == "0\nStart\n1\nMiddle\n"
    result = tmp_path / "results" / "Start_Target_FirstLinkClassifier.txt"
    assert json.loads(result.read_text()) == {
        "target_reached": True,
        "attempts": 2,
        "time_elapsed": 2.5,
        "visited_links": ["Start", "Middle", "Target"],
    }


def test_predict_stops_at_attempt_limit(capsys, monkeypatch, tmp_path) -> None:
    """Report failure after consuming the allowed attempts."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    speedrun = WikipediaSpeedrun(
        "Start", "Target", FirstLinkClassifier("Start", "Target")
    )

    with (
        patch.object(
            WikipediaSpeedrun,
            "_get_wikipedia_links",
            side_effect=[["Elsewhere"], ["Another"]],
        ) as get_links,
        patch("src.wiki_speedrun.time.time", side_effect=[10.0, 11.0]),
    ):
        speedrun.predict(max_attempts=2)

    assert get_links.call_count == 2
    assert capsys.readouterr().out == "0\nStart\n1\nElsewhere\n"
    result = tmp_path / "results" / "Start_Target_FirstLinkClassifier.txt"
    assert json.loads(result.read_text()) == {
        "target_reached": False,
        "attempts": 2,
        "time_elapsed": 1.0,
        "visited_links": ["Start", "Elsewhere", "Another"],
    }


def test_predict_replaces_previous_result(monkeypatch, tmp_path) -> None:
    """Keep the latest result for one classifier and page pair."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    speedrun = WikipediaSpeedrun(
        "Start", "Target", FirstLinkClassifier("Start", "Target")
    )

    with patch("src.wiki_speedrun.time.time", side_effect=[1.0, 2.0, 3.0, 5.0]):
        speedrun.predict(max_attempts=0)
        speedrun.predict(max_attempts=0)

    result = tmp_path / "results" / "Start_Target_FirstLinkClassifier.txt"
    assert json.loads(result.read_text())["time_elapsed"] == 2.0

"""Exercise complete races with simulated Wikipedia and Jev HTTP responses."""

import json
from email.message import Message
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from src.speedrun import JEV_API, WIKIPEDIA_API, _request, speedrun


class FakeAPI:
    """Small directed graph exposed through the actual API request shapes."""

    def __init__(self, graph, *, aliases=None, choice=None):
        self.graph = graph
        self.aliases = aliases or {}
        self.ids = {title: index for index, title in enumerate(graph, 1)}
        self.choice = choice
        self.requests = []
        self.scored = []

    def __call__(self, request):
        self.requests.append(request)
        if request.full_url == JEV_API:
            return self.jev(json.loads(request.data))
        params = parse_qs(urlsplit(request.full_url).query)
        if params["action"] == ["query"]:
            pages = {}
            for title in params["titles"][0].split("|"):
                title = self.aliases.get(title, title)
                page = {"title": title, "ns": 0}
                if title in self.ids:
                    page["pageid"] = self.ids[title]
                else:
                    page["missing"] = True
                pages[title] = page
            return {"query": {"pages": list(pages.values())}}
        pageid = int(params["pageid"][0])
        title = next(title for title, index in self.ids.items() if index == pageid)
        links = [
            {"ns": 0, "title": target, "exists": True} for target in self.graph[title]
        ]
        links.extend(
            [
                {"ns": 14, "title": "Category:Ignored", "exists": True},
                {"ns": 0, "title": "Missing", "exists": False},
            ]
        )
        return {"parse": {"pageid": pageid, "title": title, "links": links}}

    def jev(self, payload):
        """Prefer later candidates so truncation would lose the winning link."""
        questions = payload["questions"]
        if "next_page" in questions:
            options = questions["next_page"]["criteria"]
            assert len(options) <= 255
            chosen = self.choice or max(options, key=int)
            return {"answers": {"next_page": {"choice": chosen}}}
        self.scored.extend(questions)
        return {
            "answers": {
                key: {"score": 2 * int(key) / len(self.graph)} for key in questions
            }
        }


def run_fake(api, start="Start", target="Target", **kwargs):
    """Use the real navigation, extraction and selection against a fake server."""
    with (
        patch("src.speedrun._request", side_effect=api),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}),
    ):
        return speedrun(start, target, **kwargs)


@pytest.mark.parametrize(
    "graph, start, max_hops, status, path",
    [
        ({"Target": []}, "Target", 0, "success", ["Target"]),
        (
            {"Start": ["Target"], "Target": []},
            "Start",
            1,
            "success",
            ["Start", "Target"],
        ),
        ({"Start": ["Target"], "Target": []}, "Start", 0, "hop_limit", ["Start"]),
        ({"Start": [], "Target": []}, "Start", 30, "dead_end", ["Start"]),
        (
            {"Start": ["Bridge"], "Bridge": ["Start"], "Target": []},
            "Start",
            30,
            "dead_end",
            ["Start", "Bridge"],
        ),
        (
            {"Start": ["Bridge"], "Bridge": ["Target"], "Target": []},
            "Start",
            1,
            "hop_limit",
            ["Start", "Bridge"],
        ),
    ],
)
def test_navigation_boundaries(graph, start, max_hops, status, path):
    """Immediate success, exact hop budgets, forced moves, and cycles terminate."""
    result = run_fake(FakeAPI(graph), start=start, max_hops=max_hops)
    assert (result.status, result.path, result.hops) == (status, path, len(path) - 1)
    assert result.jev_calls == 0
    assert result.error is None
    assert result.elapsed_seconds >= result.wikipedia_seconds >= 0


def test_redirects_and_duplicate_links():
    """Aliases of visited pages are excluded; a target alias wins immediately."""
    api = FakeAPI(
        {"Start": ["Self", "Goal", "Goal", "Other"], "Target": [], "Other": []},
        aliases={"Self": "Start", "Goal": "Target", "Beginning": "Start"},
    )
    result = run_fake(api, start="Beginning")
    assert result.path == ["Start", "Target"]
    assert result.status == "success"
    assert result.jev_calls == 0


@pytest.mark.parametrize("count, calls", [(255, 1), (256, 4)])
def test_choice_limit_and_scoring_all_links(count, calls):
    """All candidates survive extraction; over-limit lists use score then choice."""
    titles = [f"Article {index}" for index in range(count)]
    graph = {"Start": titles, **{title: ["Target"] for title in titles}, "Target": []}
    api = FakeAPI(graph)
    result = run_fake(api)
    assert result.status == "success"
    assert result.path == ["Start", titles[-1], "Target"]
    assert result.jev_calls == calls
    assert len(set(api.scored)) == (count if count > 255 else 0)
    queries = [
        parse_qs(urlsplit(request.full_url).query)
        for request in api.requests
        if request.full_url != JEV_API
    ]
    assert all(len(query.get("titles", [""])[0].split("|")) <= 50 for query in queries)
    assert result.jev_seconds > 0


@pytest.mark.parametrize("title", ["", "  ", "Start|Target", None])
def test_invalid_titles(title):
    """Reject empty titles and API multi-title injection before requesting data."""
    with pytest.raises(ValueError):
        speedrun(title, "Target")


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_hop_limit(limit):
    """Only nonnegative integer hop limits are accepted."""
    with pytest.raises(ValueError):
        speedrun("Start", "Target", max_hops=limit)


def test_missing_page_and_invalid_choice():
    """Failures are explicit and never fabricate a traversed link."""
    missing = run_fake(FakeAPI({"Start": []}))
    assert missing.status == "error"
    assert "Not an existing" in missing.error
    api = FakeAPI({"Start": ["A", "B"], "A": [], "B": [], "Target": []}, choice="999")
    result = run_fake(api)
    assert result.status == "error"
    assert result.path == ["Start"]
    assert "outside the available links" in result.error


def test_missing_key_only_fails_when_a_decision_is_needed():
    """Direct moves work without a key; model decisions report missing config."""
    api = FakeAPI({"Start": ["A", "B"], "A": [], "B": [], "Target": []})
    with (
        patch("src.speedrun._request", side_effect=api),
        patch.dict("os.environ", {"TYPESAFE_API_KEY": ""}),
    ):
        result = speedrun("Start", "Target")
        same = speedrun("Target", "Target")
    assert "TYPESAFE_API_KEY" in result.error
    assert result.jev_calls == 0
    assert same.status == "success"


def test_http_retry_and_timeout():
    """Rate limiting honors Retry-After, with a timeout on every attempt."""
    headers = Message()
    headers["Retry-After"] = "3"
    failure = HTTPError(WIKIPEDIA_API, 429, "limited", headers, None)
    with (
        patch(
            "src.speedrun.urlopen", side_effect=[failure, BytesIO(b'{"ok":1}')]
        ) as opened,
        patch("src.speedrun.time.sleep") as sleep,
    ):
        assert _request(Request(WIKIPEDIA_API)) == {"ok": 1}
    sleep.assert_called_once_with(3.0)
    assert opened.call_count == 2
    assert all(call.kwargs["timeout"] == 15 for call in opened.call_args_list)


def test_network_failure_preserves_path():
    """Network failures surface as errors with a partial route and timing."""
    api = FakeAPI({"Start": ["Target"], "Target": []})
    resolve = api.__call__

    def fail_on_parse(request):
        if "action=parse" in request.full_url:
            raise URLError("offline")
        return resolve(request)

    result = run_fake(fail_on_parse)
    assert result.status == "error"
    assert result.path == ["Start"]
    assert "offline" in result.error


@pytest.mark.parametrize("status, attempts", [(401, 1), (422, 1), (503, 3), (529, 3)])
def test_http_failures_are_bounded(status, attempts):
    """Permanent errors stop immediately; transient errors exhaust two retries."""
    failure = HTTPError(JEV_API, status, "failed", Message(), None)
    with (
        patch("src.speedrun.urlopen", side_effect=failure) as opened,
        patch("src.speedrun.time.sleep") as sleep,
        pytest.raises(HTTPError),
    ):
        _request(Request(JEV_API))
    assert opened.call_count == attempts
    assert sleep.call_count == attempts - 1

"""Tests for the Wikipedia link classifiers."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.classifiers import (
    JevLinkClassifier,
    LinkClassifier,
    OpenAILinkClassifier,
)


class RecordingClassifier(LinkClassifier):
    """Record batch sizes without calling a model."""

    def __init__(self) -> None:
        super().__init__("Start", "Target")
        self.batch_sizes: list[int] = []

    def _select_from_batch(self, links: list[str]) -> str:
        self.batch_sizes.append(len(links))
        return links[0]


@pytest.mark.parametrize(
    ("link_count", "expected_batches"),
    [(255, [255]), (256, [255, 1, 2]), (510, [255, 255, 2])],
)
def test_classifier_batches_links_once(link_count, expected_batches) -> None:
    """Each candidate appears in one first-round batch."""
    classifier = RecordingClassifier()

    classifier.select_link([str(index) for index in range(link_count)])

    assert classifier.batch_sizes == expected_batches


def test_jev_classifier_builds_choice_from_links() -> None:
    """Jev receives every link as an option and returns its selected title."""
    client = Mock()
    client.system_one.return_value = SimpleNamespace(
        choices={"page": SimpleNamespace(choice="Beta")}
    )

    with patch("src.classifiers.TypeSafeClient", return_value=client):
        classifier = JevLinkClassifier("Start", "Target")

    assert classifier.select_link(["Alpha", "Beta"]) == "Beta"
    request = client.system_one.call_args.kwargs
    assert request["state"] == {
        "initial_page": "Start",
        "target_page": "Target",
    }
    assert request["questions"]["page"].criteria == {"Alpha": None, "Beta": None}


def test_openai_classifier_accepts_a_listed_link() -> None:
    """OpenAI's selected title is returned when it belongs to the batch."""
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Beta"))]
    )

    with patch("src.classifiers.OpenAI", return_value=client):
        classifier = OpenAILinkClassifier("Start", "Target")

    assert classifier.select_link(["Alpha", "Beta"]) == "Beta"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-4.1-mini"
    assert request["temperature"] == 0


def test_openai_classifier_uses_random_link_for_unknown_response() -> None:
    """OpenAI falls back to a valid random link after hallucinating."""
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Unknown"))]
    )

    with (
        patch("src.classifiers.OpenAI", return_value=client),
        patch("src.classifiers.random.choice", return_value="Alpha") as random_choice,
    ):
        classifier = OpenAILinkClassifier("Start", "Target")
        selected_link = classifier.select_link(["Alpha", "Beta"])

    assert selected_link == "Alpha"
    random_choice.assert_called_once_with(["Alpha", "Beta"])

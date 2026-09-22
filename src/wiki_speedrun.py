"""Wikipedia speedrun."""

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.classifiers import LinkClassifier
from src.config import config


class WikipediaSpeedrun:
    """Class to build Wikipedia speedrun."""

    def __init__(self, start: str, target: str, classifier: LinkClassifier) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        self.start = start
        self.target = target
        self.classifier = classifier

    @staticmethod
    def _get_wikipedia_links(page: str) -> list[str]:
        """Returns the existing Wikipedia article links on a page.

        Args:
            start: Starting page.

        Returns:
            All articles links on the page.
        """

        query = urlencode(
            {
                "action": "parse",
                "page": page,
                "prop": "links",
                "format": "json",
                "formatversion": 2,
            }
        )
        # Generic headers may be blocked by the Wikipedia API
        request = Request(
            f"https://en.wikipedia.org/w/api.php?{query}",
            headers={"User-Agent": "WikipediaSpeedrunJev"},
        )

        with urlopen(request, timeout=config.timeout) as response:
            data = json.load(response)

        return [
            link["title"]
            for link in data["parse"]["links"]
            if link["ns"] == 0 and link.get("exists", False)
        ]

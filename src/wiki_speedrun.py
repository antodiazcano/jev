"""Wikipedia speedrun."""

import json
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.classifiers import LinkClassifier


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
    def _get_wikipedia_links(
        page: str, max_retries: int = 3, timeout: int = 10
    ) -> list[str]:
        """Returns the existing Wikipedia article links on a page.

        Args:
            start: Starting page.
            max_retries: Maximum number of retries for a Wikipedia request.
            timeout: Timeout for the Wikipedia request.

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
            headers={
                "User-Agent": (
                    "JevWikiSpeedrunBot/0.1 " "(https://github.com/antodiazcano/jev)"
                )
            },
        )

        for attempt in range(max_retries):
            try:
                with urlopen(request, timeout=timeout) as response:
                    data = json.load(response)
                break
            except HTTPError as error:
                if error.code != 429 or attempt == 2:
                    raise
                retry_after = error.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else 5 * 2**attempt
                time.sleep(delay)

        return [
            link["title"]
            for link in data["parse"]["links"]
            if link["ns"] == 0 and link.get("exists", False)
        ]

    def _save_results(
        self,
        reached_target: bool,
        attempts: int,
        time_elapsed: float,
        visited_links: list[str],
    ) -> None:
        """Saves the results of the `predict` function in a txt file.

        Args:
            reached_target: `True` if the target was reached, `False` otherwise.
            attempts: Number of attempts.
            time_elapsed: Time elapsed executing the `predict` function.
            visited_links: Visited links.
        """

        filename = (
            (f"{self.start}__{self.target}__{type(self.classifier).__name__}.json")
            .lower()
            .replace(" ", "_")
        )
        path = f"results/{filename}"

        data = {
            "target_reached": reached_target,
            "attempts": attempts,
            "time_elapsed": round(time_elapsed, 3),
            "visited_links": visited_links,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def predict(self, max_attempts: int = 100) -> None:
        """Goes from the starting page to the target page.

        Args:
            max_attempts: Maximum allowed attempts.
        """

        current_page = self.start
        attempts = 0
        visited_links = [self.start]
        t0 = time.time()

        while current_page != self.target and attempts < max_attempts:
            print(f"{attempts}\n{current_page}\n\n")
            links = self._get_wikipedia_links(current_page)
            links = [link for link in links if link not in visited_links]
            current_page = self.classifier.select_link(links)
            visited_links.append(current_page)
            attempts += 1

        reached_target = current_page == self.target
        time_elapsed = time.time() - t0

        self._save_results(reached_target, attempts, time_elapsed, visited_links)

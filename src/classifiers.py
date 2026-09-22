"""Script to define the possible classifiers."""

import random
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI
from typesafe_sdk import Choice, TypeSafeClient


class LinkClassifier(ABC):
    """Interface for a link classifier."""

    def __init__(self, start: str, target: str) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        load_dotenv()

        self.start = start
        self.target = target

    @abstractmethod
    def _select_from_batch(self, links: list[str]) -> str:
        """Selects the most promising link of the batch for the target.

        Args:
            Links of the batch.

        Returns:
            Most promising link of the batch.
        """

    def select_link(self, links: list[str]) -> str:
        """Selects the most promising link for the target.

        Args:
            Possible links to choose.

        Returns:
            Most promising link.
        """

        n = len(links)
        batch_size = 255

        if n <= batch_size:
            return self._select_from_batch(links)

        winners = []
        for i in range(0, n, batch_size):
            winners.append(self._select_from_batch(links[i : min(n, i + batch_size)]))

        return self.select_link(winners)


class JevLinkClassifier(LinkClassifier):
    """Link classifier with Jev."""

    def __init__(self, start: str, target: str) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        super().__init__(start, target)

        self.client = TypeSafeClient(model="jev-latest")

    def _select_from_batch(self, links: list[str]) -> str:
        """Selects the most promising link of the batch for the target.

        Args:
            Links of the batch.

        Returns:
            Most promising link of the batch.
        """

        state = {"initial_page": self.start, "target_page": self.target}
        instructions = (
            "You are doing a Wikipedia Speedrun. Which of the following pages is the "
            "most promising one to arrive to the target?"
        )
        criteria = {link: None for link in links}

        question = Choice(instructions=instructions, criteria=criteria)
        response = self.client.system_one(state=state, questions={"page": question})

        return response.choices["page"].choice


class OpenAILinkClassifier(LinkClassifier):
    """Link classifier with OpenAI."""

    def __init__(self, start: str, target: str, model: str = "gpt-4.1-mini") -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
            model: OpenAI model.
        """

        super().__init__(start, target)

        self.client = OpenAI()
        self.model = model

    def _select_from_batch(self, links: list[str]) -> str:
        """Selects the most promising link of the batch for the target.

        Args:
            links: Links of the batch.

        Returns:
            Most promising link of the batch.
        """

        prompt = (
            f"You are doing a Wikipedia Speedrun. The initial page was {self.start}, "
            f"and the target page is {self.target}. Which of the following pages is "
            "the most promising one to arrive to the target? \n\nAnswer ONLY with the "
            "name that appears in the list, don't tell anything more. \nThis is the "
            f"list: \n{links}"
        )
        response = (
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0,
            )
            .choices[0]
            .message.content
        )

        if (response is None) or (response not in links):
            print("The model hallucinated!")
            return random.choice(links)

        return response

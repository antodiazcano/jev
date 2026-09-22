"""Script to define the possible classifiers."""

from abc import ABC, abstractmethod

from dotenv import load_dotenv
from groq import Groq
from typesafe_sdk import Choice, TypeSafeClient


class LinkClassifier(ABC):
    """Interface for a link classifier."""

    def __init__(self, start: str, target: str) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        self.start = start
        self.target = target

    @abstractmethod
    def select_link(self, links: list[str]) -> str:
        """Selects the most promising link for the target.

        Args:
            Possible links to choose.

        Returns:
            Most promising link.
        """


class JevLinkClassifier(LinkClassifier):
    """Link classifier with Jev."""

    def __init__(self, start: str, target: str) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        super().__init__(start, target)

        load_dotenv()
        self.client = TypeSafeClient(model="jev-1.13.0")

    def _get_prompt(self) -> str:
        """Obtains the prompt for the model.

        Returns:
            Prompt the model will use.
        """

        return (
            f"You are doing a Wikipedia Speedrun. The initial page was {self.start}, "
            f"and the target page is {self.target}. Which of the following pages is "
            "the most promising one to arrive to the target?"
        )

    def select_link(self, links: list[str]) -> str:
        """Selects the most promising link for the target.

        Args:
            Possible links to choose.

        Returns:
            Most promising link.
        """

        state = (
            f"You are doing a Wikipedia Speedrun. The initial page was {self.start}, "
            f"and the target page is {self.target}."
        )
        instructions = (
            "Which of the following pages is the most promising one to arrive to the "
            "target?"
        )
        question = Choice(instructions=instructions, criteria={})

        response = self.client.system_one(state=state, questions={"page": question})

        return str(response.choices["page"])


class GroqLinkClassifier(LinkClassifier):
    """Link classifier with Groq."""

    def __init__(
        self, start: str, target: str, model: str = "openai/gpt-oss-120b"
    ) -> None:
        """Constructor of the class.

        Args:
            start: Starting page.
            target: Target page.
        """

        super().__init__(start, target)

        load_dotenv()
        self.client = Groq()
        self.model = model

    def select_link(self, links: list[str]) -> str:
        """Selects the most promising link for the target.

        Args:
            Possible links to choose.

        Returns:
            Most promising link.

        Raises:
            ValueError: If there was an error processing the request.
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

        if response is None:
            raise RuntimeError("There was an error processing the request!")

        return response

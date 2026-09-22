"""Script to test the speedrun."""

from src.classifiers import JevLinkClassifier, OpenAILinkClassifier
from src.wiki_speedrun import WikipediaSpeedrun


def main() -> None:
    """Compares the two classifiers for the speedrun."""

    start = "Cristiano Ronaldo"
    target = "Joe Elliott"

    print("Starting Jev\n")
    speedrun_jev = WikipediaSpeedrun(start, target, JevLinkClassifier(start, target))
    speedrun_jev.predict()

    print("\n\nStarting OpenAI\n")
    speedrun_openai = WikipediaSpeedrun(
        start, target, OpenAILinkClassifier(start, target)
    )
    speedrun_openai.predict()


if __name__ == "__main__":
    main()

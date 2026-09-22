# Wikipedia Speedrun with Jev

A small Python project that races from one Wikipedia article to another by
following only links found on each page. At every step, an AI classifier chooses
the link that looks most promising.

The project compares two classifiers:

- **Jev**, using TypeSafe's `Choice` primitive.
- **OpenAI**, using `gpt-4.1-mini` by default.

## How it works

1. Fetch all valid Wikipedia article links from the current page.
2. Remove pages that have already been visited.
3. Ask the selected classifier to choose the best next link.
4. Repeat until the target is reached or the attempt limit is exhausted.

Jev accepts up to 255 choices per request, so larger link collections are split
into batches and their winners are compared recursively.

## Setup

The project requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Add your API keys to `.env`:

```dotenv
TYPESAFE_API_KEY=your-typesafe-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Run the race

Choose the starting and target pages in `src/main.py`, then run:

```bash
uv run python -m src.main
```

The script runs both classifiers and prints every visited page. Each result is
saved as JSON in `results/`, including whether the target was reached, the number
of attempts, elapsed time, and the complete route.

Example:

```json
{
    "target_reached": true,
    "attempts": 39,
    "time_elapsed": 53.566,
    "visited_links": ["Cristiano Ronaldo", "...", "Joe Elliott"]
}
```

## Tests

```bash
make test
```

Run the complete test and quality suite with:

```bash
make check
```

## Project structure

```text
src/classifiers.py       AI link classifiers and batching logic
src/wiki_speedrun.py     Wikipedia traversal and result storage
src/main.py              Example race configuration
tests/                   Unit tests
results/                 Saved race results
```

## License

This project is released under the [MIT License](LICENSE).

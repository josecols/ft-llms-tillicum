import json
from pathlib import Path

from datasets import load_dataset

SEED = 1234
SPLITS = ("train", "validation", "test")
BASE_PATH = Path(__file__).resolve().parent


def extract_abstracts(samples: dict) -> dict:
    """
    Extracts the abstracts from the given article samples.

    Args:
        samples: The batch of article samples.

    Returns: A new column with the extracted abstracts.
    """
    abstracts = [article.split("\n")[0].strip() for article in samples["article"]]

    return {"abstract": abstracts}


def format_model_messages(text: str, summary: str | None = None) -> dict:
    """
    Applies Chat Markup Language structure to an input text for lay biomedical summarization.

    Args:
        text: The input article text.
        summary: The ground-truth summary.

    Returns: A list of message dictionaries in ChatML format.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10-20 sentence summary for non-experts. The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms.",
            },
            {
                "role": "user",
                "content": text,
            },
            {
                "role": "assistant",
                "content": f"Summary:{summary or ''}",
            },
        ]
    }


def prepare_data_split(split: str, output_dir: Path) -> None:
    filename = f"plos_{split}"
    ds = (
        load_dataset("BioLaySumm/BioLaySumm2025-PLOS", split=split)
        .shuffle(seed=SEED)
    )
    ds_with_abstracts = ds.map(extract_abstracts, batched=True)
    ds_with_abstracts.to_parquet(output_dir / f"{filename}.parquet")

    output_path = output_dir / f"{filename}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in ds_with_abstracts:
            json_line = format_model_messages(
                record.get("abstract", ""), record.get("summary", "")
            )
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds_with_abstracts)} records to {output_path}")


def prepare_data():
    output_dir = (BASE_PATH / ".." / "data").resolve()
    for split in SPLITS:
        prepare_data_split(split, output_dir)


if __name__ == "__main__":
    prepare_data()

import argparse
import json
from pathlib import Path

from datasets import load_dataset

SEED = 1234
SPLITS = ("train", "validation", "test")
BASE_PATH = Path(__file__).resolve().parent


def extract_text(samples: dict, use_abstracts: bool = False) -> dict:
    """
    Extracts the text from the given article samples.

    Args:
        samples: The batch of article samples.
        use_abstracts: If True, extract only abstracts; if False, use the entire article text.

    Returns: A new column with the extracted text.
    """
    if use_abstracts:
        texts = [article.split("\n")[0].strip() for article in samples["article"]]
    else:
        texts = samples["article"]

    return {"text": texts}


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


def prepare_data_split(
    split: str, output_dir: Path, use_abstracts: bool = False
) -> None:
    filename = f"plos_{split}"
    ds = load_dataset("BioLaySumm/BioLaySumm2025-PLOS", split=split).shuffle(seed=SEED)
    ds_with_text = ds.map(
        lambda samples: extract_text(samples, use_abstracts=use_abstracts), batched=True
    )
    ds_with_text.to_parquet(output_dir / f"{filename}.parquet")

    output_path = output_dir / f"{filename}.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in ds_with_text:
            json_line = format_model_messages(
                record.get("text", ""), record.get("summary", "")
            )
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds_with_text)} records to {output_path}")


def prepare_data(use_abstracts: bool = False):
    output_dir = (BASE_PATH / ".." / "data").resolve()
    for split in SPLITS:
        prepare_data_split(split, output_dir, use_abstracts=use_abstracts)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare BioLaySumm dataset for training."
    )
    parser.add_argument(
        "--use-abstracts",
        action="store_true",
        default=False,
        help="Extract only abstracts from articles.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    prepare_data(use_abstracts=args.use_abstracts)


if __name__ == "__main__":
    main()

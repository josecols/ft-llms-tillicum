import argparse

from models import LlamaSummarizerTuned


def main(**kwargs):
    config = {**kwargs}
    adapter_path = config.pop("adapter_path")

    summarizer = LlamaSummarizerTuned(adapter_path, **config)
    summarizer.generate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on a fine-tuned Llama 3.2 3B Instruct model."
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model adapter",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where to save the generated summaries",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset to use for inference",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="text",
        help="Field name in the dataset to use as input for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()
    main(**vars(parser.parse_args()))

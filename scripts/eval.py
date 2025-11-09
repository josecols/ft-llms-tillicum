import argparse
from pprint import pprint

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


def evaluate_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    Calculate average ROUGE scores across multiple text pairs.

    Args:
        predictions: List of predicted summaries.
        references: List of ground-truth summaries.

    Returns: Average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    """

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True, split_summaries=True
    )
    # Compute the scores for all prediction/reference pairs.
    scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]

    rouge1 = np.mean([score["rouge1"].fmeasure for score in scores])
    rouge2 = np.mean([score["rouge2"].fmeasure for score in scores])
    rougeL = np.mean([score["rougeLsum"].fmeasure for score in scores])
    average = np.mean([rouge1, rouge2, rougeL])

    return {
        "rouge1": rouge1.item(),
        "rouge2": rouge2.item(),
        "rougeL": rougeL.item(),
        "average": average.item(),
    }


def load_predictions(predictions_path: str) -> list[str]:
    """
    Load predictions from a CSV file.

    Args:
        predictions_path: Path to the CSV file containing predictions.

    Returns: List of predicted summaries.
    """
    df = pd.read_csv(predictions_path)
    return df["summary"].tolist()


def load_references(dataset_path: str, reference_field: str = "summary") -> list[str]:
    """
    Load reference summaries from a parquet dataset file.

    Args:
        dataset_path: Path to the dataset file.
        reference_field: Field name in the reference summaries.

    Returns: List of reference summaries.
    """
    df = pd.read_parquet(dataset_path)
    return df[reference_field].tolist()


def main(**kwargs):
    predictions_path = kwargs["predictions_path"]
    dataset_path = kwargs["dataset_path"]

    predictions = load_predictions(predictions_path)
    references = load_references(dataset_path)
    assert len(predictions) == len(references)
    print(f"Loaded {len(predictions)} predictions.")

    print("Calculating ROUGE scores...")
    scores = evaluate_rouge(predictions, references)
    pprint(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model summaries.")

    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to the CSV file containing model predictions",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset file containing reference summaries",
    )

    args = parser.parse_args()
    main(**vars(args))

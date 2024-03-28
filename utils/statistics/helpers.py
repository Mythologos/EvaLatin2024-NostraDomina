from os import listdir
from os.path import isfile, isdir
from typing import Sequence


def gather_scorer_filepaths(input_filepath: str, subsets: list[str]) -> list[str]:
    if isfile(input_filepath) and input_filepath.endswith(".tsv"):
        filepaths: list[str] = [input_filepath]
    elif isdir(input_filepath):
        filenames: list[str] = [filename for filename in listdir(input_filepath) if filename.endswith(".tsv")]
        filepaths: list[str] = [f"{input_filepath}/{filename}" for filename in filenames]
    else:
        raise ValueError("Input filepath is not either a single TSV file or a directory of TSV files.")

    if len(subsets) > 0:
        filepaths: list[str] = list(filter(lambda path: any([subset in path for subset in subsets]), filepaths))

    return filepaths


def read_scorer_file(scorer_filepath: str) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    ground_truths: list[str] = []

    with open(scorer_filepath, encoding="utf-8", mode="r") as scorer_file:
        for line_index, line in enumerate(scorer_file):
            if line_index == 0:
                continue
            else:
                prediction, ground_truth = line.strip().split("\t")
                predictions.append(prediction)
                ground_truths.append(ground_truth)

    return predictions, ground_truths


def gather_scored_values(filepaths: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    predictions: list[list[str]] = []
    ground_truths: list[list[str]] = []
    for filepath in filepaths:
        individual_predictions, individual_ground_truths = read_scorer_file(filepath)
        predictions.append(individual_predictions)
        ground_truths.append(individual_ground_truths)

    return predictions, ground_truths


def gather_labels(predictions: list[list[str]], ground_truths: list[list[str]]) -> Sequence[str]:
    labels: set[str] = set()
    for i in range(0, len(predictions)):
        labels.update(set(predictions[i]))
        labels.update(set(ground_truths[i]))

    labels: Sequence[str] = tuple(labels)
    return labels

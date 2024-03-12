from argparse import ArgumentParser, Namespace
from math import floor
from os.path import isdir, isfile
from random import choice, seed, shuffle

from utils.cli.messages import GeneralMessage, SplitterMessage
from utils.data.annotation import POINTS
from utils.data.loaders.polarity import PolaritySentence, PolarityDataset

SplitMap = dict[str, list[PolaritySentence]]


DEFAULT_INPUT_FILEPATH: str = "data/polarity/training/annotated/polarity-coordinate/full/AutoSentimentTreebankv1.tsv"
DEFAULT_OUTPUT_FILEPATH: str = "data/polarity/training/annotated/polarity-coordinate/splits"


def generate_random_splits(data: list[PolaritySentence], names: list[str], ratios: list[float]) -> SplitMap:
    splits: SplitMap = {name: [] for name in names}
    quotas: dict[str, int] = {names[i]: floor(ratios[i] * len(data)) for i in range(0, len(names))}
    while sum(quotas.values()) < len(data):
        minimum_quota_set: str = min(list(quotas.keys()), key=lambda k: quotas[k])
        quotas[minimum_quota_set] += 1

    while len(data) > 0:
        element: PolaritySentence = data.pop(0)
        candidates: list[str] = [split for split, split_data in splits.items() if len(split_data) < quotas[split]]
        recipient: str = choice(candidates)
        splits[recipient].append(element)

    assert sum([len(values) for values in splits.values()]) == sum([value for value in quotas.values()])
    return splits


def save_splits(output_directory: str, splits: SplitMap):
    split_statistics: dict[str, dict[str, int]] = {}
    for split in splits.keys():
        with open(f"{output_directory}/{split}.tsv", encoding="utf-8", mode="w+") as split_file:
            split_statistics[split] = {class_name: 0 for class_name in POINTS}
            for sentence in splits[split]:
                split_statistics[split][sentence.attributes["polarity"]] += 1
                distances: dict[str, float] = sentence.attributes["distances"]
                output_distances: str = ";".join([str(distances[class_name]) for class_name in POINTS])
                split_file.write(f"{sentence.sentence_id}\t"
                                 f"{sentence.sentence_text}\t"
                                 f"{sentence.attributes['polarity']}\t"
                                 f"{output_distances}\n")

    statistics_output: str = "We save a dataset with the following statistics:"
    for key, values in split_statistics.items():
        statistics_output += f"\n\t* {key}:"
        for class_name, class_quantity in values.items():
            statistics_output += f"\n\t\t** {class_name}: {class_quantity}"
    else:
        statistics_output += f"\nTotals: "
        for class_name in POINTS.keys():
            statistics_output += f"\n\t* {class_name}: " \
                                 f"{sum([split_statistics[split][class_name] for split in split_statistics.keys()])}"

    print(statistics_output)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--input-file", type=str, default=DEFAULT_INPUT_FILEPATH, help=SplitterMessage.INPUT_FILE)
    parser.add_argument(
        "--names", type=str, nargs="+", default=["training", "validation", "test"], help=SplitterMessage.NAMES
    )
    parser.add_argument(
        "--output-directory", type=str, default=DEFAULT_OUTPUT_FILEPATH, help=SplitterMessage.OUTPUT_DIRECTORY
    )
    parser.add_argument("--random-seed", type=int, default=42, help=GeneralMessage.RANDOM_SEED)
    parser.add_argument("--ratios", type=float, nargs="+", default=[.8, .1, .1], help=SplitterMessage.RATIOS)
    parser.add_argument("--strategy", type=str, default="random", choices=("random",), help=SplitterMessage.STRATEGY)
    args: Namespace = parser.parse_args()

    seed(args.random_seed)

    if not isfile(args.input_file):
        raise ValueError(f"The input file, <{args.input_file}>, is not a valid file.")
    elif not isdir(args.output_directory):
        raise ValueError(f"The output directory, <{args.output_directory}>, is not a valid directory.")

    assert len(args.names) == len(args.ratios)

    polarity_sentences: list[PolaritySentence] = PolarityDataset.load_polarity_file([args.input_file])
    shuffle(polarity_sentences)

    if args.strategy == "random":
        data_splits: dict[str, list[PolaritySentence]] = \
            generate_random_splits(polarity_sentences, args.names, args.ratios)
    else:
        raise ValueError(f"The strategy <{args.strategy}> is not recognized.")

    save_splits(args.output_directory, data_splits)

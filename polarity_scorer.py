from argparse import ArgumentParser, Namespace
from statistics import mean
from typing import Sequence

from sklearn.metrics import f1_score

from utils.cli.messages import ResultsMessage
from utils.statistics.helpers import gather_scorer_filepaths, gather_scored_values

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--input-filepath", type=str, required=True, help=ResultsMessage.INPUT_FILEPATH)
    parser.add_argument(
        "--subsets", type=str, nargs="*", choices=["Orazio", "Pontano", "Seneca"], default=[],
        help=ResultsMessage.SUBSETS
    )
    args: Namespace = parser.parse_args()

    all_filepaths: list[str] = gather_scorer_filepaths(args.input_filepath, args.subsets)
    all_predictions, all_ground_truths = gather_scored_values(all_filepaths)

    macro_f1_scores: list[float] = []
    micro_f1_scores: list[float] = []
    for i in range(0, len(all_predictions)):
        assert len(all_ground_truths[i]) == len(all_predictions[i])
        labels: Sequence[str] = \
            tuple(set([label for label in all_ground_truths[i]]).union(set([label for label in all_predictions[i]])))
        macro_f1: float = \
            f1_score(all_ground_truths[i], all_predictions[i], average="macro", labels=labels, zero_division=0)
        micro_f1: float = \
            f1_score(all_ground_truths[i], all_predictions[i], average="micro", labels=labels, zero_division=0)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)

    average_macro_f1: float = mean(macro_f1_scores)
    average_micro_f1: float = mean(micro_f1_scores)
    for i in range(0, len(macro_f1_scores)):
        filename: str = all_filepaths[i].split("/")[-1]
        print(f"{all_filepaths[i]}:"
              f"\n\t* Macro-F1 Score: {macro_f1_scores[i]:.2f}"
              f"\n\t* Micro-F1 Score: {micro_f1_scores[i]:.2f}"
              f"\n")

    print(f"Average Macro-F1 Score: {average_macro_f1:.2f}\n"
          f"Average Micro-F1 Score: {average_micro_f1:.2f}")

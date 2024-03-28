from argparse import ArgumentParser, Namespace
from typing import Any, Sequence

from natsort import natsorted

from utils.cli.messages import ResultsMessage, VisualizerMessage
from utils.statistics.helpers import gather_scored_values, gather_labels, gather_scorer_filepaths
from utils.visualization.visualizers import visualize_confusion_matrix

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--label-order", type=str, nargs="*", default=["positive", "negative", "neutral", "mixed"],
        help=VisualizerMessage.LABEL_ORDER
    )
    parser.add_argument("--input-filepath", type=str, required=True, help=ResultsMessage.INPUT_FILEPATH)
    parser.add_argument("--output-filepath", type=str, help=VisualizerMessage.OUTPUT_FILEPATH)
    parser.add_argument(
        "--subsets", type=str, nargs="*", choices=["Orazio", "Pontano", "Seneca"], default=[],
        help=ResultsMessage.SUBSETS
    )
    args: Namespace = parser.parse_args()

    all_filepaths: list[str] = gather_scorer_filepaths(args.input_filepath, args.subsets)
    all_predictions, all_ground_truths = gather_scored_values(all_filepaths)
    if args.label_order is not None:
        all_labels = args.label_order
    else:
        all_labels: Sequence[str] = gather_labels(all_predictions, all_ground_truths)
        all_labels = natsorted(all_labels)

    confusion_matrix_kwargs: dict[str, Any] = {
        "output_filepath": args.output_filepath,
        "subtitle": "All" if len(args.subsets) == 0 else "; ".join(args.subsets)
    }
    visualize_confusion_matrix(all_predictions, all_ground_truths, all_labels, **confusion_matrix_kwargs)

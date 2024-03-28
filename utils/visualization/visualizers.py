from typing import Sequence

from matplotlib import pyplot
from matplotlib.colors import Colormap
from matplotlib.pyplot import colormaps
from numpy import int16, zeros
from numpy.typing import NDArray


def visualize_confusion_matrix(predictions: list[list[str]], ground_truths: list[list[str]],
                               labels: Sequence[str], **kwargs):
    assert len(predictions) == len(ground_truths)
    confusion_matrix: NDArray = zeros((len(labels), len(labels)), dtype=int16)
    for i in range(0, len(predictions)):
        assert len(predictions[i]) == len(ground_truths[i])
        for j in range(0, len(predictions[i])):
            prediction_class_index: int = labels.index(predictions[i][j])
            ground_truth_class_index: int = labels.index(ground_truths[i][j])
            confusion_matrix[prediction_class_index, ground_truth_class_index] += 1

    label_range: range = range(0, len(labels))
    mean_value: float = confusion_matrix.mean().item()

    pyplot.rcParams["font.sans-serif"] = ["TeX Gyre Heros"]
    figure, axis = pyplot.subplots()
    colormap: Colormap = colormaps["Purples"]
    axis.imshow(confusion_matrix, cmap=colormap)
    axis.set_xticks(label_range, rotation=45, ha="right", rotation_mode="anchor", labels=labels, fontsize=16)
    axis.set_yticks(label_range, labels=labels, fontsize=16)
    for source_index in label_range:
        for destination_index in label_range:
            if confusion_matrix[destination_index, source_index] < mean_value:
                color: str = "k"
            else:
                color = "w"

            confusion_matrix_value: int = confusion_matrix[destination_index, source_index].item()
            axis.text(
                source_index, destination_index, str(confusion_matrix_value),
                ha="center", va="center", color=color, fontsize=15
            )

    axis.set_title(f"Confusion Matrix [{kwargs['subtitle']}]", fontsize=18)
    pyplot.ylabel("Predictions", fontsize=16)
    pyplot.xlabel("Ground Truths", fontsize=16)
    pyplot.tight_layout()

    if kwargs.get("output_filepath", None) is not None:
        pyplot.savefig(f"{kwargs['output_filepath']}.pdf", bbox_inches="tight")
    else:
        pyplot.show()

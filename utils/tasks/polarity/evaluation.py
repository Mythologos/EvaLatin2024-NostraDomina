from sys import stdout
from typing import Any, Optional, TextIO, Union

from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix
from torch import argmax, inference_mode, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models.bases import NeuralClassifier
from utils.tasks.common.constants import OutputDict


@inference_mode()
def evaluate_classifier(evaluating_model: NeuralClassifier, evaluation_dataset: DataLoader,
                        evaluation_kwargs: dict[str, Any], evaluation_file: TextIO) -> OutputDict:
    evaluating_model.eval()

    outputs: OutputDict = {
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None
    }

    tqdm_bool: bool = not evaluation_kwargs["tqdm"]
    predictions: list[str] = []
    ground_truth_values: list[str] = []
    for batch, classes, _, batch_kwargs in tqdm(evaluation_dataset, file=stdout, disable=tqdm_bool):
        # We turn the input sentences and tags into tensors.
        log_probabilities: Tensor = evaluating_model(batch, **batch_kwargs)
        sentence_predictions: list[int] = argmax(log_probabilities, dim=1).tolist()  # (B, C) -> (B).
        named_predictions: list[str] = evaluating_model.revert_classes(sentence_predictions)
        predictions.extend(named_predictions)
        ground_truth_values.extend(classes)

    labels: list[str] = list(evaluating_model.vocabularies["class_to_index"].keys())
    output_report: dict[str, dict[str, Union[int, float]]] = \
        classification_report(ground_truth_values, predictions, labels=labels, zero_division=0.0, output_dict=True)
    output_matrix: NDArray = confusion_matrix(ground_truth_values, predictions, labels=labels).tolist()
    outputs["precision"] = output_report["macro avg"]["precision"]
    outputs["recall"] = output_report["macro avg"]["recall"]
    outputs["f1"] = output_report["macro avg"]["f1-score"]
    outputs["confusion_matrix"] = output_matrix
    write_evaluation_outputs(evaluation_file, outputs, labels)
    return outputs


def write_evaluation_outputs(evaluation_file: Optional[TextIO], evaluation_outputs: OutputDict, labels: list[str]):
    if evaluation_file is not None:
        confusion_matrix_rows: list[str] = ["\t\t" + ("\t\t".join(labels))]
        for i in range(0, len(labels)):
            middle_row_items: list[str] = [labels[i]]
            for j in range(0, len(labels)):
                middle_row_items.append(str(evaluation_outputs["confusion_matrix"][i][j]))
            confusion_matrix_rows.append("\t\t".join(middle_row_items))

        confusion_matrix_output: str = "\n".join(confusion_matrix_rows)
        evaluation_file.write(f"Overall Classification Results:"
                              f"\n\t* Macro-Averaged Precision: {evaluation_outputs['precision']}"
                              f"\n\t* Macro-Averaged Recall: {evaluation_outputs['recall']}"
                              f"\n\t* Macro-Averaged F1: {evaluation_outputs['f1']}"
                              f"\n\nConfusion Matrix "
                              f"({len(labels)} [Ground Truth] x {len(labels)} [Predicted]):"
                              f"\n{confusion_matrix_output}\n")

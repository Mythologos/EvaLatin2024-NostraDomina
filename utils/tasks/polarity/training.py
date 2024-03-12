from math import inf
from sys import stdout
from time import time
from typing import Any, Optional, TextIO, Type

from torch import save, Tensor
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models.bases import NeuralClassifier
from utils.tasks.common.constants import OutputDict
from utils.tasks.io import FileDict
from utils.tasks.polarity.evaluation import evaluate_classifier


def train_classifier(training_model: NeuralClassifier, dataset: dict[str, DataLoader], training_kwargs: dict[str, Any],
                     evaluation_kwargs: dict[str, Any], file_kwargs: FileDict) -> OutputDict:
    outputs: OutputDict = {
        "best_epoch_count": 0,   # This refers to the count of the epoch, starting from 1.
        "best_epoch_index": 0,  # This refers to the actual 0-based index for the epoch.
        "best_f1_score": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_confusion_matrix": None,
        "dev_f1s": [],
        "dev_precisions": [],
        "dev_recalls": [],
        "epoch_durations": [],
        "epoch_losses": [],
        "training_f1s": [],
        "training_precisions": [],
        "training_recalls": [],
        "total_loss": 0.0
    }

    training_file: Optional[TextIO] = file_kwargs["training_file"]
    validation_file: Optional[TextIO] = file_kwargs["evaluation_file"]

    if file_kwargs.get("model_location", None) is not None:
        save(training_model, file_kwargs["model_location"])

    optimizer: Type[Optimizer] = training_kwargs["optimizer"]
    optimizer_kwargs: dict[str, Any] = training_kwargs["optimizer_kwargs"]
    optimizer: Optimizer = optimizer(training_model.parameters(), **optimizer_kwargs)

    loss_function: _Loss = training_kwargs["loss_function"]

    tqdm_bool: bool = not training_kwargs["tqdm"]
    current_epoch: int = 1
    current_patience: int = 0
    while current_epoch <= training_kwargs["epochs"] and current_patience < training_kwargs["patience"]:
        epoch_start: float = time()
        print(f"Starting epoch {current_epoch}...", flush=True)

        if file_kwargs["training_file"] is not None:
            file_kwargs["training_file"].write(f"Starting epoch {current_epoch}...\n")

        training_model.train()

        epoch_loss: float = 0.0
        for batch, classes, distances, batch_kwargs in tqdm(dataset["training"], file=stdout, disable=tqdm_bool):
            # We turn the input sentences and tags into tensors.
            classes_tensor: Tensor = training_model.prepare_classes(classes)
            distances_tensor: Optional[Tensor] = training_model.prepare_distances(distances)
            log_probabilities: Tensor = training_model(batch, **batch_kwargs)
            batch_loss: Tensor = loss_function(log_probabilities, classes_tensor, distance_weights=distances_tensor)

            # We set our gradient back to zero.
            optimizer.zero_grad()
            # We back-propagate the loss.
            batch_loss.backward()
            # We clip gradients.
            clip_grad_norm_(training_model.parameters(), 1.0)
            # We perform a step with the optimizer.
            optimizer.step()

            # We store the detached batch loss to track the epoch loss.
            epoch_loss += batch_loss.detach().item()

        epoch_end: float = time()

        print(f"Finishing epoch {current_epoch}. Validating...", flush=True)

        if file_kwargs["training_file"] is not None:
            file_kwargs["training_file"].write(f"Finishing epoch {current_epoch}. Validating...\n")

        outputs["epoch_durations"].append(epoch_end - epoch_start)
        outputs["epoch_losses"].append(epoch_loss)
        outputs["total_loss"] += epoch_loss

        # Next, we evaluate the model on both the training set and the validation set.
        if training_kwargs["training_interval"] != inf and \
                (current_epoch - 1) % training_kwargs["training_interval"] == 0:
            training_results: dict[str, Any] = \
                evaluate_classifier(training_model, dataset["training"], evaluation_kwargs, training_file)

            outputs["training_precisions"].append(training_results["precision"])
            outputs["training_recalls"].append(training_results["recall"])
            outputs["training_f1s"].append(training_results["f1"])

            if training_file is not None:
                training_file.write("\n")

        if validation_file is not None:
            validation_file.write(f"Getting results for Epoch {current_epoch}...\n")

        validation_results: dict[str, Any] = \
            evaluate_classifier(training_model, dataset["evaluation"], evaluation_kwargs, validation_file)

        if validation_file is not None:
            validation_file.write("\n")

        outputs["dev_precisions"].append(validation_results["precision"])
        outputs["dev_recalls"].append(validation_results["recall"])
        outputs["dev_f1s"].append(validation_results["f1"])

        if outputs["best_f1_score"] == 0.0 or validation_results["f1"] >= outputs["best_f1_score"]:
            print(f"We save a model with the following results:"
                  f"\n\t* Precision: {validation_results['precision']}"
                  f"\n\t* Recall: {validation_results['recall']}"
                  f"\n\t* F1: {validation_results['f1']}"
                  f"\n\t* Confusion Matrix:"
                  f"\n{validation_results['confusion_matrix']}")

            outputs["best_epoch_count"] = current_epoch
            outputs["best_epoch_index"] = current_epoch - 1   # Since current_epoch starts at 1, we subtract 1 to index.
            outputs["best_f1_score"] = validation_results["f1"]
            outputs["best_precision"] = validation_results["precision"]
            outputs["best_recall"] = validation_results["recall"]
            outputs["best_confusion_matrix"] = validation_results["confusion_matrix"]

            if file_kwargs.get("model_location", None) is not None:
                save(training_model, file_kwargs["model_location"])

            # We add a condition to allow for the "most recent" model to be saved, even if the best score didn't change.
            # This also prevents a model from running to its maximum number of epochs if it doesn't pick up at all
            #   after a certain amount of time.
            if outputs["best_f1_score"] > 0.0:
                current_patience = 0   # We reset the patience, since the last model did better.
            else:
                current_patience += 1
        else:
            print(f"We do NOT save a model with the following results:"
                  f"\n\t* Precision: {validation_results['precision']}"
                  f"\n\t* Recall: {validation_results['recall']}"
                  f"\n\t* F1: {validation_results['f1']}"
                  f"\n\t* Confusion Matrix:"
                  f"\n{validation_results['confusion_matrix']}")

            current_patience += 1   # We increment the patience.

        # We prepare for the next epoch...
        current_epoch += 1

    return outputs

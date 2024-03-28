from os.path import isdir
from sys import stdout
from typing import Any, Optional, TextIO, Union

from torch import argmax, inference_mode, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data.loaders.polarity import PolaritySentence
from utils.models.bases import NeuralClassifier


PredictionMapping = dict[str, list[tuple[PolaritySentence, str, Optional[str]]]]


@inference_mode()
def gather_classifier_predictions(predicting_model: NeuralClassifier, prediction_dataset: DataLoader,
                                  prediction_kwargs: dict[str, Any], prediction_filepath: str):
    predicting_model.eval()

    tqdm_bool: bool = not prediction_kwargs["tqdm"]
    sentences: list[PolaritySentence] = []
    predictions: list[str] = []
    ground_truths: list[Optional[str]] = []
    for batch, classes, _, batch_kwargs in tqdm(prediction_dataset, file=stdout, disable=tqdm_bool):
        # We turn the input sentences and tags into tensors.
        log_probabilities: Tensor = predicting_model(batch, **batch_kwargs)
        sentence_predictions: list[int] = argmax(log_probabilities, dim=1).tolist()  # (B, C) -> (B).
        named_predictions: list[str] = predicting_model.revert_classes(sentence_predictions)
        sentences.extend(batch_kwargs["sentences"])
        predictions.extend(named_predictions)
        ground_truths.extend(classes)

    write_predictions(
        prediction_filepath, sentences, predictions, ground_truths, prediction_kwargs["prediction_format"]
    )


def write_predictions(prediction_filepath: Optional[Union[str, TextIO]], sentences: list[PolaritySentence],
                      predictions: list[str], ground_truths: list[Optional[str]], prediction_format: str):
    sorted_predictions: PredictionMapping = sort_predictions(sentences, predictions, ground_truths)
    if prediction_format == "full":
        write_full_tsv(prediction_filepath, sorted_predictions)
    elif prediction_format == "scorer":
        write_scorer_tsv(prediction_filepath, sorted_predictions)
    else:
        raise ValueError(f"The prediction format <{prediction_format}> is not recognized.")


def write_full_tsv(prediction_filepath: str, sorted_predictions: PredictionMapping):
    if prediction_filepath is not None:
        if isdir(prediction_filepath) is True:
            for source, predicted_sentences in sorted_predictions.items():
                *_, source_filename = source.split("/")
                with open(f"{prediction_filepath}/{source_filename}", mode="w+", encoding="utf-8") as output_tsv_file:
                    output_string: str = ""
                    for (sentence, prediction, _) in predicted_sentences:
                        output_string += f"{sentence.sentence_id}\t{sentence.sentence_text}\t{prediction}\n"
                    else:
                        output_tsv_file.write(output_string)
        else:
            raise ValueError(f"The filepath <{prediction_filepath}> is not a valid directory.")
    else:
        output_string: str = ""
        for source, predicted_sentences in sorted_predictions.items():
            output_string += f"Results for <{source}>:"
            for (sentence, prediction, _) in predicted_sentences:
                output_string += f"\n\t* {sentence.attributes['line']}, {sentence.sentence_id}: " \
                                 f"{sentence.sentence_text[0]} ... {sentence.sentence_text[-1]} ;; " \
                                 f"{prediction}"
            else:
                output_string += "\n\n"
        else:
            stdout.write(output_string)


def write_scorer_tsv(prediction_filepath: str, sorted_predictions: PredictionMapping):
    if prediction_filepath is not None:
        if isdir(prediction_filepath) is True:
            for source, predicted_sentences in sorted_predictions.items():
                *_, source_filename = source.split("/")
                with open(f"{prediction_filepath}/{source_filename}", mode="w+", encoding="utf-8") as output_tsv_file:
                    output_string: str = "P\tGS\n"
                    for (sentence, prediction, ground_truth) in predicted_sentences:
                        output_string += f"{prediction}\t{ground_truth}\n" if ground_truth is not None \
                            else f"{prediction}\t\n"
                    else:
                        output_tsv_file.write(output_string)
        else:
            raise ValueError(f"The filepath <{prediction_filepath}> is not a valid directory.")
    else:
        output_string: str = ""
        for source, predicted_sentences in sorted_predictions.items():
            output_string += f"Results for <{source}>:"
            output_string: str = "P\tGS\n"
            for (sentence, prediction, ground_truth) in predicted_sentences:
                output_string += f"{prediction}\t{ground_truth}\n" if ground_truth is not None \
                    else f"{prediction}\t\n"
            else:
                output_string += "\n\n"
        else:
            stdout.write(output_string)


def sort_predictions(sentences: list[PolaritySentence], predictions: list[str], ground_truths: list[Optional[str]]) -> \
        PredictionMapping:
    predictions_by_file: PredictionMapping = {}
    for i in range(0, len(sentences)):
        sentence: PolaritySentence = sentences[i]
        prediction: str = predictions[i]
        ground_truth: Optional[str] = ground_truths[i]
        predicted_sentence: tuple[PolaritySentence, str, Optional[str]] = (sentence, prediction, ground_truth)

        source: str = sentence.attributes["source"]
        if predictions_by_file.get(source, None) is None:
            predictions_by_file[source] = []
        predictions_by_file[source].append(predicted_sentence)

    for source, predicted_sentences in predictions_by_file.items():
        predictions_by_file[source] = sorted(predicted_sentences, key=lambda p: p[0].attributes["line"])

    return predictions_by_file

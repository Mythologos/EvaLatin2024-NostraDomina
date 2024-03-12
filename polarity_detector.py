from argparse import Namespace
from json import dump
from random import seed
from typing import Any, Optional, TextIO, Union

from torch import device, manual_seed, load, Generator
from torch.utils.data import DataLoader

from utils.cli.helpers import add_common_optional_arguments, add_common_training_arguments, setup_parser_divisions, \
    set_device, add_classification_training_arguments, add_classification_inference_arguments, resolve_filepaths, \
    add_classification_prediction_arguments
from utils.cli.messages import DetectorMessage
from utils.data.annotation import CLASSES
from utils.data.loaders.polarity import LatinLMCollator, NamedPolarityDataset, load_dataset
from utils.data.tokenizers import get_tokenizer
from utils.layers.embeddings import NamedEmbedding
from utils.layers.encoders import NamedEncoder
from utils.models.bases import NamedArchitecture, NeuralClassifier
from utils.models.interface import build_model
from utils.tasks.common.constants import OutputDict
from utils.tasks.io import FileDict, define_file_kwargs
from utils.tasks.looping import define_evaluation_loop_kwargs, define_training_loop_kwargs
from utils.tasks.loss import NamedLossFunction
from utils.tasks.polarity.evaluation import evaluate_classifier
from utils.tasks.polarity.prediction import gather_classifier_predictions
from utils.tasks.polarity.training import train_classifier


DEFAULT_ARCHITECTURE: NamedArchitecture = NamedArchitecture.NEURAL_CLASSIFIER


def determine_collator_options(mode: str, embedding_type: str, loss_function_type: Optional[str] = None) -> \
        dict[str, Any]:
    options: dict[str, Any] = {}
    if embedding_type in (NamedEmbedding.CANINE_C, NamedEmbedding.CANINE_S):
        options["should_pretokenize"] = False
    else:
        options["should_pretokenize"] = True

    if mode == "train":
        if loss_function_type == NamedLossFunction.NLL:
            options["distance_type"] = None
        elif loss_function_type == NamedLossFunction.GDW_NLL:
            options["distance_type"] = "gold"
        else:
            raise ValueError(f"Loss function <{loss_function_type}> not recognized.")

    return options


def write_training_results(files: FileDict, outputs: OutputDict):
    if files.get("model_output_location", None) is not None:
        model_output_file: TextIO = open(files["model_output_location"], encoding="utf-8", mode="w+")
        dump(outputs, model_output_file, indent=1)
        model_output_file.close()


def print_evaluation_results(outputs: OutputDict):
    print(f"The model completed with the following results:"
          f"\n\t* Precision: {outputs['precision']}"
          f"\n\t* Recall: {outputs['recall']}"
          f"\n\t* F1: {outputs['f1']}"
          f"\n\t* Confusion Matrix:"
          f"\n{outputs['confusion_matrix']}\n")


if __name__ == "__main__":
    parser, training_subparser, evaluation_subparser, prediction_subparser = setup_parser_divisions()
    add_common_training_arguments(training_subparser)
    add_classification_training_arguments(training_subparser)
    add_classification_prediction_arguments(prediction_subparser)

    for subparser in (evaluation_subparser, prediction_subparser):
        add_classification_inference_arguments(subparser)

    for subparser in (training_subparser, evaluation_subparser, prediction_subparser):
        subparser.add_argument(
            "--dataset", type=str, choices=list(NamedPolarityDataset), required=True, help=DetectorMessage.DATASET
        )
        add_common_optional_arguments(subparser)

    args: Namespace = parser.parse_args()
    kwargs: dict[str, Any] = vars(args)

    default_device: device = set_device()   # Determines and sets default device for all tensors.

    # We provide a random seed to make computations deterministic.
    print(f"Running with random seed <{args.random_seed}> on device <{default_device}> ...", flush=True)
    seed(args.random_seed)
    manual_seed(args.random_seed)
    generator: Generator = Generator(device=default_device).manual_seed(args.random_seed)

    components: dict[str, Union[NamedEmbedding, NamedEncoder]] = {"embedding": args.embedding, "encoder": args.encoder}
    label_vocabularies: dict[str, dict[str, Any]] = {
        "class_to_index": CLASSES,
        "index_to_class": {value: key for key, value in CLASSES.items()}
    }

    resolve_filepaths(kwargs)   # Sets up appropriate paths for loading pre-trained model and tokenizer.
    tokenizer, subword_vocabulary, maximum_sequence_length = get_tokenizer(components["embedding"], **kwargs)
    collator: LatinLMCollator = LatinLMCollator(tokenizer, subword_vocabulary, maximum_length=maximum_sequence_length)

    loading_kwargs: dict[str, Any] = {
        "common": {"batch_size": args.batch_size, "collate_fn": collator},
        "generator": generator,
        "inference_split": None if args.mode == "train" else args.inference_split
    }
    dataset: dict[str, DataLoader] = load_dataset(args.dataset, args.mode, loading_kwargs)

    file_kwargs: FileDict = define_file_kwargs(kwargs)
    evaluation_kwargs: dict[str, Any] = define_evaluation_loop_kwargs(kwargs)

    if args.mode == "train":
        model: NeuralClassifier = build_model(DEFAULT_ARCHITECTURE, components, label_vocabularies, tokenizer, kwargs)
        training_kwargs: dict[str, Any] = define_training_loop_kwargs(kwargs)
        loss_function_name, _ = args.loss_function
        collator_options: dict[str, Any] = \
            determine_collator_options(args.mode, args.embedding, loss_function_name)
        collator.set_collator_options(**collator_options)
        training_outputs: OutputDict = train_classifier(model, dataset, training_kwargs, evaluation_kwargs, file_kwargs)
        write_training_results(file_kwargs, training_outputs)
    elif args.mode in ("evaluate", "predict"):
        model: NeuralClassifier = load(file_kwargs["model_location"])
        collator_options: dict[str, Any] = \
            determine_collator_options(args.mode, model.components["embedding"])
        collator.set_collator_options(**collator_options)

        if args.mode == "evaluate":
            evaluation_file: TextIO = file_kwargs["evaluation_file"]
            evaluation_outputs: OutputDict = \
                evaluate_classifier(model, dataset["evaluation"], evaluation_kwargs, evaluation_file)
            print_evaluation_results(evaluation_outputs)
        else:
            prediction_filepath: str = file_kwargs["results_location"]
            evaluation_kwargs["prediction_format"] = args.prediction_format
            gather_classifier_predictions(model, dataset["evaluation"], evaluation_kwargs, prediction_filepath)
    else:
        raise ValueError(f"The mode <{args.mode}> is not recognized")

from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Sequence

from torch import cuda, device, set_default_device

from utils.cli.messages import DetectorMessage, GeneralMessage
from utils.data.tokenizers import DEFAULT_TOKENIZER_FILEPATHS
from utils.layers.embeddings import DEFAULT_EMBEDDING_FILEPATHS, NamedEmbedding
from utils.layers.encoders import NamedEncoder
from utils.tasks.looping import get_optimizer
from utils.tasks.loss import get_classification_loss_function, NamedLossFunction


DEFAULT_SPLITS: list[str] = ["training", "validation", "test"]


def setup_parser_divisions() -> Sequence[ArgumentParser]:
    main_parser: ArgumentParser = ArgumentParser()
    subparsers = main_parser.add_subparsers(title="mode", dest="mode", required=True)
    train_parser: ArgumentParser = subparsers.add_parser("train")
    evaluate_parser: ArgumentParser = subparsers.add_parser("evaluate")
    predict_parser: ArgumentParser = subparsers.add_parser("predict")
    return main_parser, train_parser, evaluate_parser, predict_parser


def add_common_optional_arguments(parser_group):
    parser_group.add_argument("--batch-size", type=int, default=1, help=DetectorMessage.BATCH_SIZE)
    parser_group.add_argument(
        "--embedding", required=True, type=str, choices=list(NamedEmbedding), help=DetectorMessage.EMBEDDING
    )
    parser_group.add_argument(
        "--encoder", required=True, type=str, choices=list(NamedEncoder), help=DetectorMessage.ENCODER
    )
    parser_group.add_argument(
        "--evaluation-filename", nargs="?", type=str, default=None, help=DetectorMessage.EVALUATION_FILENAME
    )
    parser_group.add_argument("--model-location", type=str, default=None, help=GeneralMessage.MODEL_LOCATION)
    parser_group.add_argument("--model-name", nargs="?", type=str, default=None, help=GeneralMessage.MODEL_NAME)
    parser_group.add_argument(
        "--pretrained-filepath", type=str, default="auto", help=DetectorMessage.PRETRAINED_FILEPATH
    )
    parser_group.add_argument("--random-seed", type=int, default=42, help=GeneralMessage.RANDOM_SEED)
    parser_group.add_argument(
        "--results-location", nargs="?", type=str, default=None, help=GeneralMessage.RESULTS_LOCATION
    )
    parser_group.add_argument("--tokenizer-filepath", type=str, default="auto", help=DetectorMessage.TOKENIZER_FILEPATH)
    parser_group.add_argument("--tqdm", action=BooleanOptionalAction, default=True, help=DetectorMessage.TQDM)

    # Since the option for --tqdm is to *disable* it, the boolean obtained here gets flipped in practice.


def add_common_training_arguments(training_group):
    # At least one of the following is also required:
    training_group.add_argument("--epochs", type=int, default=None, help=DetectorMessage.EPOCHS)
    training_group.add_argument("--patience", type=int, default=None, help=DetectorMessage.PATIENCE)

    # Optional Arguments:
    training_group.add_argument(
        "--bidirectional", action=BooleanOptionalAction, default=False, help=DetectorMessage.BIDIRECTIONAL
    )
    training_group.add_argument(
        "--frozen-embeddings", action=BooleanOptionalAction, default=True, help=DetectorMessage.FROZEN_EMBEDDINGS
    )
    training_group.add_argument("--heads", "--num-heads", nargs="?", type=int, default=1, help=DetectorMessage.HEADS)
    training_group.add_argument("--hidden-size", type=int, default=100, help=DetectorMessage.HIDDEN_SIZE)
    training_group.add_argument("--layers", "--num-layers", nargs="?", type=int, default=1, help=DetectorMessage.LAYERS)
    training_group.add_argument(
        "--lr", "--learning-rate", nargs="?", type=float, default=0.01, help=DetectorMessage.LEARNING_RATE
    )
    training_group.add_argument(
        "--optimizer", nargs="?", type=get_optimizer, default="Adam", help=DetectorMessage.OPTIMIZER
    )
    training_group.add_argument("--output-filename", type=str, default="training", help=DetectorMessage.OUTPUT_FILENAME)
    training_group.add_argument("--output-location", type=str, default="results", help=DetectorMessage.OUTPUT_LOCATION)
    training_group.add_argument(
        "--training-filename", nargs="?", type=str, default=None, help=GeneralMessage.TRAINING_FILENAME
    )
    training_group.add_argument(
        "--training-interval", nargs="?", type=str, default="inf", help=GeneralMessage.TRAINING_INTERVAL
    )


def add_classification_training_arguments(training_group):
    training_group.add_argument(
        "--loss-function", type=get_classification_loss_function, default=NamedLossFunction.NLL,
        help=DetectorMessage.LOSS_FUNCTION
    )


def add_classification_inference_arguments(inference_group):
    inference_group.add_argument(
        "--inference-split", type=str, choices=DEFAULT_SPLITS, default="test", help=GeneralMessage.INFERENCE_SPLIT
    )


def add_classification_prediction_arguments(prediction_group):
    prediction_group.add_argument(
        "--prediction-format", type=str, choices=("full", "scorer"), default="full",
        help=DetectorMessage.PREDICTION_FORMAT
    )


def set_device() -> device:
    torch_device: device = device("cuda") if cuda.is_available() else device("cpu")
    set_default_device(device=torch_device)
    return torch_device


def retrieve_default_filepath(named_embedding: NamedEmbedding, key: str, table: dict[NamedEmbedding, str]) -> str:
    try:
        default_filepath: str = table[named_embedding]
    except KeyError:
        raise ValueError(f"The embedding <{named_embedding}> is not present in the table "
                         f"corresponding to key <{key}> ")
    return default_filepath


def resolve_filepaths(kwargs: dict[str, Any]):
    for key in ("pretrained_filepath", "tokenizer_filepath"):
        if kwargs.get(key, None) is None:
            raise ValueError(f"<{key}> was not defined.")
        elif kwargs[key] == "auto":
            if key == "pretrained_filepath":
                table: dict[NamedEmbedding, str] = DEFAULT_EMBEDDING_FILEPATHS
            elif key == "tokenizer_filepath":
                table = DEFAULT_TOKENIZER_FILEPATHS
            else:
                raise ValueError(f"<{key}> for embedding-related filepaths not recognized.")

            kwargs[key] = retrieve_default_filepath(kwargs["embedding"], key, table)
        else:
            continue

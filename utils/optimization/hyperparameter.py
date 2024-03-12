from argparse import ArgumentParser, Action, Namespace
from typing import Any, NamedTuple, Optional, Sequence, TypeVar, Type, Union

from numpy import concatenate, iinfo, int32, linspace

from utils.constants import NamedEnum
from utils.data.loaders.polarity import NamedPolarityDataset
from utils.layers.embeddings import NamedEmbedding
from utils.layers.encoders import NamedEncoder
from utils.tasks.loss import NamedLossFunction

T = TypeVar('T')
HyperparameterRange = Sequence[T]

BOOLEAN_ACTION_RANGE: tuple[str, str] = ("True", "False")

BATCH_RANGE: HyperparameterRange[int] = tuple([2 ** i for i in range(3, 8)])
BIDIRECTIONAL_RANGE: HyperparameterRange[str] = ("--bidirectional", "--no-bidirectional")
DATASET_RANGE: HyperparameterRange[str] = tuple([dataset.value for dataset in NamedPolarityDataset])
EMBEDDING_RANGE: HyperparameterRange[str] = tuple([embedding.value for embedding in NamedEmbedding])
ENCODER_RANGE: HyperparameterRange[str] = tuple([encoder.value for encoder in NamedEncoder])
EPOCHS_RANGE: HyperparameterRange[int] = tuple([10 * i for i in range(1, 11)])
FROZEN_RANGE: HyperparameterRange[str] = ("--frozen-embeddings", "--no-frozen-embeddings")
HEADS_RANGE: HyperparameterRange[int] = tuple([2 ** i for i in range(0, 5)])
HIDDEN_SIZE_RANGE: HyperparameterRange[int] = (64, 96, 128, 192, 256, 384, 512, 768, 1024)
LAYERS_RANGE: HyperparameterRange[int] = tuple(range(1, 5))
LEARNING_RATE_RANGE: HyperparameterRange[float] = concatenate(
    (linspace(0.00001, 0.0001, num=10), linspace(0.0001, 0.001, num=10),
     linspace(0.001, 0.01, num=10)), axis=0
)
LOSS_FUNCTION_RANGE: HyperparameterRange[str] = tuple([loss_function.value for loss_function in NamedLossFunction])
OPTIMIZER_RANGE: HyperparameterRange[str] = \
    ("ASGD", "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD")
PATIENCE_RANGE: HyperparameterRange[int] = tuple([1, 5, 10, 15, 20, 25])
RANDOM_SEED_RANGE: HyperparameterRange[int] = linspace(0, iinfo(int32).max)


class NamedHyperparameter(NamedEnum):
    BATCH_SIZE: str = "batch-size"
    BIDIRECTIONAL: str = "bidirectional"
    DATASET: str = "dataset"
    EMBEDDING: str = "embedding"
    ENCODER: str = "encoder"
    EPOCHS: str = "epochs"
    FROZEN_EMBEDDINGS: str = "frozen-embeddings"
    HEADS: str = "heads"
    HIDDEN_SIZE: str = "hidden-size"
    LAYERS: str = "layers"
    LEARNING_RATE: str = "learning-rate"
    LOSS_FUNCTION: str = "loss-function"
    OPTIMIZER: str = "optimizer"
    PATIENCE: str = "patience"
    RANDOM_SEED: str = "random-seed"


class Hyperparameter(NamedTuple):
    name: str
    range: HyperparameterRange[T]
    default: Union[T, bool]   # bool is permitted only when the option is an action
    mode: str
    action: bool = False


HYPERPARAMETERS: dict[str, Hyperparameter] = {
    NamedHyperparameter.BATCH_SIZE.value: Hyperparameter(
        NamedHyperparameter.BATCH_SIZE.value, BATCH_RANGE, 32, "common"
    ),
    NamedHyperparameter.BIDIRECTIONAL.value: Hyperparameter(
        NamedHyperparameter.BIDIRECTIONAL.value, BIDIRECTIONAL_RANGE, True, "training", True
    ),
    NamedHyperparameter.DATASET.value: Hyperparameter(
        NamedHyperparameter.DATASET.value, DATASET_RANGE, "coordinate-treebank", "common"
    ),
    NamedHyperparameter.EMBEDDING.value: Hyperparameter(
        NamedHyperparameter.EMBEDDING.value, EMBEDDING_RANGE, NamedEmbedding.LATIN_BERT.value, "common"
    ),
    NamedHyperparameter.ENCODER.value: Hyperparameter(
        NamedHyperparameter.ENCODER.value, ENCODER_RANGE, NamedEncoder.IDENTITY.value, "common"
    ),
    NamedHyperparameter.EPOCHS.value: Hyperparameter(
        NamedHyperparameter.EPOCHS.value, EPOCHS_RANGE, 50, "training"
    ),
    NamedHyperparameter.FROZEN_EMBEDDINGS.value: Hyperparameter(
        NamedHyperparameter.FROZEN_EMBEDDINGS.value, FROZEN_RANGE, True, "training", True
    ),
    NamedHyperparameter.HEADS.value: Hyperparameter(
        NamedHyperparameter.HEADS.value, HEADS_RANGE, 8, "training"
    ),
    NamedHyperparameter.HIDDEN_SIZE.value: Hyperparameter(
        NamedHyperparameter.HIDDEN_SIZE.value, HIDDEN_SIZE_RANGE, 128, "training"
    ),
    NamedHyperparameter.LAYERS.value: Hyperparameter(
        NamedHyperparameter.LAYERS.value, LAYERS_RANGE, 1, "training"
    ),
    NamedHyperparameter.LEARNING_RATE.value: Hyperparameter(
        NamedHyperparameter.LEARNING_RATE.value, LEARNING_RATE_RANGE, .00002, "training"
    ),
    NamedHyperparameter.LOSS_FUNCTION.value: Hyperparameter(
        NamedHyperparameter.LOSS_FUNCTION.value, LOSS_FUNCTION_RANGE, NamedLossFunction.NLL.value, "training"
    ),
    NamedHyperparameter.PATIENCE.value: Hyperparameter(
        NamedHyperparameter.PATIENCE.value, PATIENCE_RANGE, 10, "training"
    ),
    NamedHyperparameter.RANDOM_SEED.value: Hyperparameter(
        NamedHyperparameter.RANDOM_SEED.value, RANDOM_SEED_RANGE, 42, "common"
    )
}


class HyperparameterParseAction(Action):
    def __init__(self, option_strings, dest, nargs, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, hyperparameter_parser: ArgumentParser, namespace: Namespace,
                 values: Optional[Union[str, Sequence]], option_string=None, *hyper_args, **hyper_kwargs):
        specified_hyperparameters: dict[str, Any] = {}

        if len(values) % 2 != 0:
            raise ValueError("The number of values in the list is not even. "
                             "Currently, this method only supports one-to-one hyperparameter-argument pairs.")

        value_index: int = 0
        while value_index < len(values):
            hyperparameter_name: str = values[value_index]
            try:
                hyperparameter: Hyperparameter = HYPERPARAMETERS[values[value_index]]
            except KeyError:
                raise ValueError(f"The hyperparameter <{hyperparameter_name}> is not recognized.")

            if hyperparameter.action is True:
                if values[value_index + 1] not in BOOLEAN_ACTION_RANGE:
                    raise ValueError(f"The value <{values[value_index + 1]}> is not True or False for "
                                     f"hyperparameter <{hyperparameter.name}>")
                else:
                    hyperparameter_value: bool = True if values[value_index + 1] == "True" else False
            else:   # hyperparameter.action is False
                hyperparameter_default: T = hyperparameter.default
                hyperparameter_default_type: Type = type(hyperparameter_default)
                try:
                    hyperparameter_value: T = hyperparameter_default_type(values[value_index + 1])
                except ValueError:
                    raise ValueError(f"The value <{values[value_index + 1]}> cannot be converted to "
                                     f"type <{hyperparameter_default_type}>.")
                else:
                    if hyperparameter_value not in hyperparameter.range:
                        raise ValueError(f"The value <{hyperparameter_value}> is not in the defined range for "
                                         f"hyperparameter <{hyperparameter.name}>.")

            specified_hyperparameters[hyperparameter_name] = hyperparameter_value
            value_index += 2

        setattr(namespace, self.dest, specified_hyperparameters)


def handle_training_constraints(common_hyperparameters: dict[str, Any], training_hyperparameters: dict[str, Any]):
    if common_hyperparameters[NamedHyperparameter.ENCODER.value] == NamedEncoder.IDENTITY.value:
        del training_hyperparameters[NamedHyperparameter.BIDIRECTIONAL.value]
        del training_hyperparameters[NamedHyperparameter.HEADS.value]
        del training_hyperparameters[NamedHyperparameter.HIDDEN_SIZE.value]
        del training_hyperparameters[NamedHyperparameter.LAYERS.value]
    elif common_hyperparameters[NamedHyperparameter.ENCODER.value] == NamedEncoder.LSTM.value:
        del training_hyperparameters[NamedHyperparameter.HEADS.value]
    elif common_hyperparameters[NamedHyperparameter.ENCODER.value] == NamedEncoder.TRANSFORMER.value:
        del training_hyperparameters[NamedHyperparameter.BIDIRECTIONAL.value]
    else:
        raise ValueError(f"Encoder <{common_hyperparameters[NamedHyperparameter.ENCODER.value]}> not handled.")

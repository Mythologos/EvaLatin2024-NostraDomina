from typing import Type, Union

from .classifier import NeuralClassifier
from ...constants import NamedEnum


NeuralArchitecture = Union[NeuralClassifier]


class NamedArchitecture(NamedEnum):
    NEURAL_CLASSIFIER: str = "neural-classifier"


ARCHITECTURES: dict[NamedArchitecture, Type[NeuralArchitecture]] = {
    NamedArchitecture.NEURAL_CLASSIFIER: NeuralClassifier,
}


def get_architecture(name: NamedArchitecture) -> Type[NeuralArchitecture]:
    try:
        named_architecture: Type[NeuralArchitecture] = ARCHITECTURES[name]
    except KeyError:
        raise ValueError(f"The architecture named <{name}> is not recognized.")
    return named_architecture

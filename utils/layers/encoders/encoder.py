from abc import abstractmethod
from typing import Callable

from torch import Tensor
from torch.nn import Module


class EncoderLayer(Module):
    def __init__(self, component: str, input_size: int, hidden_size: int, layers: int, output_size: int,
                 extractor: Callable, **kwargs):
        super().__init__()
        self.component = component
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.output_size = output_size
        self.extractor = extractor

    @abstractmethod
    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def get_extractor(self, extractor_name: str) -> Callable:
        if extractor_name == "single":
            extractor: Callable = self._extract_single_representation
        elif extractor_name == "sequence":
            extractor: Callable = self._extract_sequence_representation
        else:
            raise ValueError(f"The extractor <{extractor_name}> is not supported.")

        return extractor

    @abstractmethod
    def _extract_single_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _extract_sequence_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

from abc import abstractmethod

from torch import Tensor
from torch.nn import Module


class EmbeddingLayer(Module):
    def __init__(self, component: str, embedding_size: int):
        super().__init__()
        self.component = component
        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

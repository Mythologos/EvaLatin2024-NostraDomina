from typing import Callable

from torch import Tensor

from utils.layers.encoders.encoder import EncoderLayer


class IdentityEncoder(EncoderLayer):
    def __init__(self, component: str, input_size: int, hidden_size: int, layers: int, extractor_type: str):
        extractor: Callable = self.get_extractor(extractor_type)
        super().__init__(component, input_size, hidden_size, layers, input_size, extractor)

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        encodings: Tensor = self.extractor(embeddings, **kwargs)   # (B, E) [Single] or (B, N, E) [Sequence]
        return encodings

    def _extract_single_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        return sequence_encodings[:, 0, :]   # (B, N, E) -> (B, E)

    def _extract_sequence_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        return sequence_encodings

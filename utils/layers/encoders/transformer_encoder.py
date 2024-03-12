from typing import Any, Callable

from torch import Tensor
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer

from utils.layers.encoders.encoder import EncoderLayer
from utils.layers.modules import PositionalEncoding


class TFEncoder(EncoderLayer):
    def __init__(self, component: str, input_size: int, hidden_size: int, layers: int, heads: int,
                 extractor_type: str):
        extractor: Callable = self.get_extractor(extractor_type)
        super().__init__(component, input_size, hidden_size, layers, input_size, extractor)
        self.heads = heads
        self.positional_encoder = PositionalEncoding(self.input_size)

        encoder_layer_kwargs: dict[str, Any] = {
            "d_model": self.input_size,
            "nhead": self.heads,
            "dim_feedforward": self.hidden_size,
            "batch_first": True,
            "norm_first": True,   # PreNorm, following Nguyen and Salazar 2019.
        }
        encoder_layer: TransformerEncoderLayer = TransformerEncoderLayer(**encoder_layer_kwargs)
        final_norm: LayerNorm = LayerNorm(input_size)
        self.encoder: TransformerEncoder = \
            TransformerEncoder(encoder_layer, num_layers=self.layers, norm=final_norm, enable_nested_tensor=False)

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:   # (B, N, E) -> (B, N, H)
        positioned_embeddings: Tensor = self.positional_encoder(embeddings)   # (B, N, E) -> (B, N, E)
        transformer_encodings: Tensor = \
            self.encoder(positioned_embeddings, src_key_padding_mask=kwargs["padding_mask"])  # (B, N, E) -> (B, N, E)
        encodings: Tensor = self.extractor(transformer_encodings, **kwargs)   # (B, E) [Single] or (B, N, E) [Sequence]
        return encodings

    def _extract_single_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        class_encodings: Tensor = sequence_encodings[:, 0, :]   # (B, N, E) -> (B, E)
        return class_encodings

    def _extract_sequence_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        return sequence_encodings

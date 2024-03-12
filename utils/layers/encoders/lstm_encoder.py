from typing import Any, Callable

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM
from torch import stack, Tensor

from utils.layers.encoders.encoder import EncoderLayer


class LSTMEncoder(EncoderLayer):
    def __init__(self, component: str, input_size: int, hidden_size: int, layers: int, extractor_type: str,
                 bidirectional: bool = False):
        extractor: Callable = self.get_extractor(extractor_type)
        super().__init__(component, input_size, hidden_size, layers, hidden_size, extractor)
        self.is_bidirectional: bool = bidirectional

        directional_hidden_size: int = self.hidden_size if self.is_bidirectional is False else self.hidden_size // 2
        encoder_kwargs: dict[str, Any] = {
            "batch_first": True,
            "input_size": self.input_size,
            "hidden_size": directional_hidden_size,
            "num_layers": self.layers,
            "bidirectional": self.is_bidirectional
        }
        self.encoder: LSTM = LSTM(**encoder_kwargs)

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:   # (B, N, E) -> (B, H)
        packed_embeddings: PackedSequence = \
            pack_padded_sequence(embeddings, kwargs["sequence_lengths"], batch_first=True, enforce_sorted=False)

        all_encodings, _ = self.encoder(packed_embeddings)   # (B, N, E) -> (B, N, H)
        unpacked_encodings, _ = pad_packed_sequence(all_encodings, batch_first=True)
        # (B, H) [Single] or (B, N, H) [Sequence]
        encodings: Tensor = self.extractor(unpacked_encodings, lengths=kwargs["sequence_lengths"], **kwargs)

        return encodings

    def _extract_single_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        # TODO: can this be done more efficiently? I suspect there's a better tensor operation via torch.
        individual_encodings: list[Tensor] = [
            sequence_encodings[i, kwargs["sequence_lengths"][i] - 1, :]
            for i in range(len(kwargs["sequence_lengths"]))
        ]   # (B, N, H) -> list[(H)]
        encodings: Tensor = stack(individual_encodings, dim=0)   # list[(H)] -> (B, H)
        return encodings

    def _extract_sequence_representation(self, sequence_encodings: Tensor, **kwargs) -> Tensor:
        return sequence_encodings

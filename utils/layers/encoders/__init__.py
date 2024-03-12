from typing import Type

from .encoder import EncoderLayer
from .identity_encoder import IdentityEncoder
from .lstm_encoder import LSTMEncoder
from .transformer_encoder import TFEncoder
from ...constants import NamedEnum


class NamedEncoder(NamedEnum):
    IDENTITY: str = "identity"
    LSTM: str = "lstm"
    TRANSFORMER: str = "transformer"


ENCODERS: dict[NamedEncoder, Type[EncoderLayer]] = {
    NamedEncoder.IDENTITY: IdentityEncoder,
    NamedEncoder.LSTM: LSTMEncoder,
    NamedEncoder.TRANSFORMER: TFEncoder,
}


def get_encoder(name: NamedEncoder) -> Type[EncoderLayer]:
    try:
        named_encoder: Type[EncoderLayer] = ENCODERS[name]
    except KeyError:
        raise ValueError(f"The encoder named <{name}> is not recognized.")
    return named_encoder

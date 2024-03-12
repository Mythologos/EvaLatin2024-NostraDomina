from typing import Any, NamedTuple, Type, Union

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from transformers import PreTrainedTokenizer

from utils.data.tokenizers.tokens import SpecialToken


LatinSubwordTokenizer = Union[SubwordTextEncoder, PreTrainedTokenizer]


class LatinLMTokenizer(NamedTuple):
    subword_tokenizer: LatinSubwordTokenizer
    special_tokens: Type[SpecialToken]
    encoding_kwargs: dict[str, Any]

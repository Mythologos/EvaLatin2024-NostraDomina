from typing import Union

from utils.constants import NamedEnum


MINIMUM_LATIN_CODEPOINT: int = 0x0021
MAXIMUM_LATIN_CODEPOINT: int = 0x0400

WORD2VEC_UNK_TOKEN: str = "<unknown>"


class BERTSpecialToken(NamedEnum):
    CLASS: str = "[CLS]"
    SEPARATION: str = "[SEP]"
    PADDING: str = "[PAD]"
    UNKNOWN: str = "[UNK]"
    MASK: str = "[MASK]"


class RoBERTaSpecialToken(NamedEnum):
    CLASS: str = "<s>"
    SEPARATION: str = "</s>"
    PADDING: str = "<pad>"
    UNKNOWN: str = "<unk>"
    MASK: str = "<mask>"


SpecialToken = Union[BERTSpecialToken, RoBERTaSpecialToken]


LATIN_BERT_PRIMITIVES: dict[str, int] = {
    BERTSpecialToken.PADDING: 0,
    BERTSpecialToken.UNKNOWN: 1,
    BERTSpecialToken.CLASS: 2,
    BERTSpecialToken.MASK: 3,
    BERTSpecialToken.SEPARATION: 4
}

# Padding is always 0, which is fine because it's the null unicode point.
# The remaining code points are private use codepoints; see https://en.wikipedia.org/wiki/Private_Use_Areas.
CANINE_NAMED_PRIMITIVES: dict[str, int] = {
    BERTSpecialToken.PADDING: 0,
    BERTSpecialToken.CLASS: 0xE000,
    BERTSpecialToken.SEPARATION: 0xE001,
    BERTSpecialToken.MASK: 0xE003
}

CANINE_INDEXED_PRIMITIVES: dict[int, str] = {value: key for key, value in CANINE_NAMED_PRIMITIVES.items()}

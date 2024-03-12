from typing import Any, Sequence, Type

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from transformers import PreTrainedTokenizer, BertTokenizer, CanineTokenizer, RobertaTokenizer

from .tokenizer import LatinLMTokenizer, LatinSubwordTokenizer
from .tokens import LATIN_BERT_PRIMITIVES, SpecialToken, BERTSpecialToken, RoBERTaSpecialToken
from utils.layers.embeddings import NamedEmbedding


DEFAULT_TOKENIZER_FILEPATHS: dict[NamedEmbedding, str] = {
    NamedEmbedding.CANINE_C: "resources/canine-c",
    NamedEmbedding.CANINE_S: "resources/canine-s",
    NamedEmbedding.LATIN_BERT: "resources/latin-bert/subword_tokenizer_latin/latin.subword.encoder",
    NamedEmbedding.MULTILINGUAL_BERT: "resources/mbert",
    NamedEmbedding.LABERTA: "resources/laberta",
    NamedEmbedding.PHILBERTA: "resources/philberta",
    NamedEmbedding.SPHILBERTA: "resources/sphilberta"
}


# For some reason, LaBERTa and PhilBERTa's models are 514 as opposed to 512, but the model seems not to work with 514.
DEFAULT_SEQUENCE_LENGTHS: dict[NamedEmbedding, int] = {
    NamedEmbedding.CANINE_C: 2048,
    NamedEmbedding.CANINE_S: 2048,
    NamedEmbedding.LATIN_BERT: 512,
    NamedEmbedding.MULTILINGUAL_BERT: 512,
    NamedEmbedding.LABERTA: 512,
    NamedEmbedding.PHILBERTA: 512,
    NamedEmbedding.SPHILBERTA: 128
}


DEFAULT_SPECIAL_TOKENIZERS: Sequence[NamedEmbedding] = \
    (NamedEmbedding.LATIN_BERT, NamedEmbedding.MULTILINGUAL_BERT, NamedEmbedding.CANINE_C, NamedEmbedding.CANINE_S)
ROBERTA_SPECIAL_TOKENIZERS: Sequence[NamedEmbedding] = \
    (NamedEmbedding.LABERTA, NamedEmbedding.PHILBERTA, NamedEmbedding.SPHILBERTA)

HF_TOKENIZERS: Sequence[NamedEmbedding] = (
    NamedEmbedding.MULTILINGUAL_BERT, NamedEmbedding.LABERTA, NamedEmbedding.PHILBERTA,
    NamedEmbedding.CANINE_C, NamedEmbedding.CANINE_S, NamedEmbedding.SPHILBERTA
)

HF_BERT_TOKENIZERS: Sequence[NamedEmbedding] = \
    (NamedEmbedding.MULTILINGUAL_BERT, NamedEmbedding.LABERTA, NamedEmbedding.PHILBERTA, NamedEmbedding.SPHILBERTA)


def get_tokenizer(embedding_name: NamedEmbedding, **kwargs) -> tuple[LatinLMTokenizer, dict[str, int], int]:
    subword_tokenizer, subword_vocabulary = load_tokenizer(embedding_name, kwargs["tokenizer_filepath"])
    special_token_enumeration: Type[SpecialToken] = load_special_tokens(embedding_name)
    encoding_kwargs: dict[str, Any] = {"add_special_tokens": False} \
        if embedding_name != NamedEmbedding.LATIN_BERT else {}
    latin_lm_tokenizer: LatinLMTokenizer = \
        LatinLMTokenizer(subword_tokenizer, special_token_enumeration, encoding_kwargs)
    maximum_sequence_length: int = load_sequence_length(embedding_name)
    return latin_lm_tokenizer, subword_vocabulary, maximum_sequence_length


# noinspection PyProtectedMember
def load_tokenizer(embedding_name: NamedEmbedding, filepath: str) -> tuple[LatinSubwordTokenizer, dict[str, int]]:
    if embedding_name == NamedEmbedding.LATIN_BERT:
        subword_tokenizer: SubwordTextEncoder = SubwordTextEncoder(filepath)

        special_token_count: int = len(LATIN_BERT_PRIMITIVES.keys())
        for key, value in subword_tokenizer._subtoken_string_to_id.items():
            subword_tokenizer._subtoken_string_to_id[key] = value + special_token_count

        for token, value in LATIN_BERT_PRIMITIVES.items():
            subword_tokenizer._subtoken_string_to_id[token] = value

        subword_vocabulary = subword_tokenizer._subtoken_string_to_id
    elif embedding_name in HF_BERT_TOKENIZERS:
        if embedding_name == NamedEmbedding.MULTILINGUAL_BERT:
            tokenizer_class: Type[PreTrainedTokenizer] = BertTokenizer
        else:
            tokenizer_class: Type[PreTrainedTokenizer] = RobertaTokenizer

        subword_tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(filepath)
        subword_vocabulary = subword_tokenizer.get_vocab()
    elif embedding_name in (NamedEmbedding.CANINE_C, NamedEmbedding.CANINE_S):
        # We do not need to make use of the pretrained filepaths, since CANINE's tokenizer is the same regardless.
        subword_tokenizer: PreTrainedTokenizer = CanineTokenizer()
        subword_vocabulary = subword_tokenizer._special_codepoints   # type: ignore
    else:
        raise ValueError(f"The embedding <{embedding_name}> is not currently recognized.")

    return subword_tokenizer, subword_vocabulary


def load_special_tokens(embedding_name: NamedEmbedding) -> Type[SpecialToken]:
    if embedding_name in DEFAULT_SPECIAL_TOKENIZERS:
        enumeration: Type[BERTSpecialToken] = BERTSpecialToken
    elif embedding_name in ROBERTA_SPECIAL_TOKENIZERS:
        enumeration: Type[RoBERTaSpecialToken] = RoBERTaSpecialToken
    else:
        raise ValueError(f"The embedding <{embedding_name}> is not currently recognized.")
    return enumeration


def load_sequence_length(embedding_name: NamedEmbedding) -> int:
    try:
        sequence_length: int = DEFAULT_SEQUENCE_LENGTHS[embedding_name]
    except KeyError:
        raise ValueError(f"The embedding <{embedding_name}> is not currently recognized.")

    return sequence_length

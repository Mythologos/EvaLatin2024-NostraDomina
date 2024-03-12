from typing import Type

from .embedding import EmbeddingLayer
from .latin_lm_embedding import LatinLMEmbedding
from .latin_sentence_lm_embedding import LatinSentenceLMEmbedding
from ...constants import NamedEnum


class NamedEmbedding(NamedEnum):
    CANINE_C: str = "canine-c"
    CANINE_S: str = "canine-s"
    LATIN_BERT: str = "latin-bert"
    LABERTA: str = "laberta"
    MULTILINGUAL_BERT: str = "mbert"
    PHILBERTA: str = "philberta"
    SPHILBERTA: str = "sphilberta"


EMBEDDINGS: dict[NamedEmbedding, Type[EmbeddingLayer]] = {
    NamedEmbedding.CANINE_C: LatinLMEmbedding,
    NamedEmbedding.CANINE_S: LatinLMEmbedding,
    NamedEmbedding.LATIN_BERT: LatinLMEmbedding,
    NamedEmbedding.LABERTA: LatinLMEmbedding,
    NamedEmbedding.MULTILINGUAL_BERT: LatinLMEmbedding,
    NamedEmbedding.PHILBERTA: LatinLMEmbedding,
    NamedEmbedding.SPHILBERTA: LatinSentenceLMEmbedding
}

DEFAULT_EMBEDDING_FILEPATHS: dict[NamedEmbedding, str] = {
    NamedEmbedding.CANINE_C: "resources/canine-c",
    NamedEmbedding.CANINE_S: "resources/canine-s",
    NamedEmbedding.LATIN_BERT: "resources/latin-bert/latin_bert",
    NamedEmbedding.MULTILINGUAL_BERT: "resources/mbert",
    NamedEmbedding.LABERTA: "resources/laberta",
    NamedEmbedding.PHILBERTA: "resources/philberta",
    NamedEmbedding.SPHILBERTA: "resources/sphilberta"
}


def get_embedding(name: NamedEmbedding) -> Type[EmbeddingLayer]:
    try:
        named_embedding: Type[EmbeddingLayer] = EMBEDDINGS[name]
    except KeyError:
        raise ValueError(f"The embedding named <{name}> is not recognized.")
    return named_embedding

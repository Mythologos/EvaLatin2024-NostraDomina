from cltk.alphabet.lat import JVReplacer
from numpy import zeros
from numpy.typing import NDArray
from gensim.models import KeyedVectors, Word2Vec

from utils.data.tokenizers.tokens import WORD2VEC_UNK_TOKEN


def load_embeddings(embedding_filepath: str) -> dict[str, NDArray]:
    replacer: JVReplacer = JVReplacer()
    latin_word2vec: KeyedVectors = Word2Vec.load(embedding_filepath).wv
    word_embeddings: dict[str, NDArray] = {}

    # As in Burns et al. 2021, we preprocess the word2vec keys to reduce ambiguity between u/v and i/j.
    #   In this way, more lemmas can be captured, as we reduce variation.
    for (key, index) in latin_word2vec.key_to_index.items():
        processed_key: str = replacer.replace(key) if key != WORD2VEC_UNK_TOKEN else key

        # If there was an embedding for the original, we keep that. Otherwise, we add the replaced version.
        if processed_key not in word_embeddings:
            word_embeddings[processed_key] = latin_word2vec[index]
    else:
        if WORD2VEC_UNK_TOKEN not in word_embeddings:
            embedding_size, *_ = latin_word2vec[0].shape
            word_embeddings[WORD2VEC_UNK_TOKEN] = zeros(embedding_size, dtype="float32")

    return word_embeddings

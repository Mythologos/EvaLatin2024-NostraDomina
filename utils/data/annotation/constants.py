from typing import Type, Union

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EmbeddingTable: Type = dict[str, NDArray]
EmbeddingType: Type = Union[EmbeddingTable, SentenceTransformer]


DEFAULT_TREEBANK_NAME: str = "AutoSentimentTreebankv1"
EMBEDDING_PATH: str = "resources/word2vec/latin_w2v_bamman_lemma300_100_1"
LEXICON_PATH: str = "data/polarity/LatinAffectusv4.tsv"
MODEL_FILEPATH: str = "models/gaussian_best.model"
PREDICTIONS_FILEPATH: str = "predictions/emotion_NostraDomina_1/scorer"
TEST_FILEPATH: str = "data/polarity/evalatin-test"

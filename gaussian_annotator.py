from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import inf
from os import environ
from os.path import isdir, isfile
from random import randrange, seed
from platform import system
from statistics import mean
from typing import Any, Optional, Sequence, Type, Union
from warnings import filterwarnings

from cltk.lemmatize import LatinBackoffLemmatizer
from numpy import array, concatenate, int16, mean, stack, zeros, argmax, split
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from torch import inference_mode
from tqdm import tqdm

from utils.cli.messages import AnnotatorMessage, GaussianMessage, GeneralMessage
from utils.data.annotation import PolarityClass, CLASSES, PolarityCoordinate
from utils.data.annotation.classes import INDEXED_CLASSES
from utils.data.annotation.helpers import construct_sentence_ids, collect_input_filepaths, gather_treebank_sentences, \
    get_output_filepath, get_polarity_lexicon, output_polarity_tsv, report_statistics, compute_polarity_coordinate
from utils.data.loaders.general.conll_loader import ConllSentence, MULTIWORD_ID_MARKER
from utils.data.loaders.general.embedding_loader import load_embeddings
from utils.data.loaders.polarity import PolarityLexicon, PolarityDataset, PolaritySentence
from utils.data.tokenizers.tokens import WORD2VEC_UNK_TOKEN

# We conditionally set an environment variable.
if system() == "Windows":
    environ["OMP_NUM_THREADS"] = "1"

EmbeddingTable: Type = dict[str, NDArray]
EmbeddingType: Type = Union[EmbeddingTable, SentenceTransformer]


TREEBANK_INPUT_PATH: str = "data/polarity/training/unannotated"
EMBEDDING_PATH: str = "resources/word2vec/latin_w2v_bamman_lemma300_100_1"
LEXICON_PATH: str = "data/polarity/LatinAffectusv4.tsv"
TREEBANK_OUTPUT_PATH: str = "data/polarity/training/annotated/gaussian/full"
SEED_FILEPATH: str = "data/polarity/GoldStandardv1-Horace.tsv"

EVALUATION_BATCH_SIZE: int = 64


COVARIANCE_TYPES: Sequence[str] = ("full", "tied", "diag", "spherical")
INITIALIZATION_STRATEGIES: Sequence[str] = ("random_from_data", "kmeans", "k-means++")
NUMBER_INITIALIZATIONS: tuple[int] = (10,)
MAX_ITERATIONS: tuple[int] = (100,)

MAXIMUM_SIZE: int = 2**32 - 1


def create_lexical_word_embedding(lemma: str, embeddings: EmbeddingTable, lexicon: PolarityLexicon) -> NDArray:
    # We get the relevant word embedding...
    word_embedding: NDArray = embeddings[lemma] if lemma in embeddings else embeddings[WORD2VEC_UNK_TOKEN]

    # We gather polarity coordinate information for the given word ...
    lexicon_score: float = lexicon[lemma]["score"] if lemma in lexicon else 0.0
    polarity: float = (lexicon_score // 2) + .5
    intensity: float = (abs(lexicon_score) // 2) + .5
    sentiment_array: NDArray = array([polarity, intensity])

    # We combine the word embedding and sentiment information.
    lexical_word_embedding: NDArray = concatenate((word_embedding, sentiment_array), axis=0)
    return lexical_word_embedding


@inference_mode()
def create_sentence_embedding(sentence: str, embeddings: SentenceTransformer, lemmata: list[str],
                              lexicon: PolarityLexicon) -> NDArray:
    # We get the relevant word embedding...
    embeddings.eval()
    sentence_embedding: NDArray = embeddings.encode([sentence]).squeeze(0)

    # We gather polarity coordinate information for the given sentence ...
    filtered_lemmata: list[Optional[str]] = []
    for lemma in lemmata:
        lexical_lemma: Optional[str] = None
        if lemma in lexicon:
            lexical_lemma = lemma
        filtered_lemmata.append(lexical_lemma)

    polarity_coordinate: PolarityCoordinate = \
        compute_polarity_coordinate(filtered_lemmata, lexicon, is_lexicon_sensitive=True)
    sentiment_array: NDArray = \
        array([polarity_coordinate.polarity, polarity_coordinate.intensity], dtype=sentence_embedding.dtype)

    # We combine the sentence embedding and sentiment information.
    lexicalized_embedding: NDArray = concatenate((sentence_embedding, sentiment_array), axis=0)
    return lexicalized_embedding


def gather_aggregated_sentence_embedding(lemmata: list[str], embeddings: EmbeddingTable,
                                         lexicon: PolarityLexicon) -> NDArray:
    word_embeddings: list[NDArray] = []
    for lemma in lemmata:
        word_embedding: NDArray = create_lexical_word_embedding(lemma, embeddings, lexicon)
        word_embeddings.append(word_embedding)
    else:
        stacked_word_embeddings: NDArray = stack(word_embeddings)
        mean_embedding: NDArray = mean(stacked_word_embeddings, axis=0)

    return mean_embedding


def load_labeled_embeddings(polarity_dataset: PolarityDataset, embeddings: EmbeddingType,
                            lexicon: PolarityLexicon, lemmatizer: LatinBackoffLemmatizer) -> tuple[NDArray, NDArray]:
    separated_sentence_embeddings: list[NDArray] = []
    labels: NDArray = zeros((len(polarity_dataset)), dtype=int16)
    for i in range(0, len(polarity_dataset)):
        sentence: PolaritySentence = polarity_dataset[i]
        words: list[str] = sentence.sentence_text.split()
        lemmata: list[Optional[str]] = [lemma for (word, lemma) in lemmatizer.lemmatize(words)]
        if isinstance(embeddings, dict) is True:
            separated_sentence_embedding: NDArray = \
                gather_aggregated_sentence_embedding(lemmata, embeddings, lexicon)
        elif isinstance(embeddings, SentenceTransformer) is True:
            separated_sentence_embedding: NDArray = \
                create_sentence_embedding(sentence.sentence_text, embeddings, lemmata, lexicon)
        else:
            raise ValueError(f"The embedding of type <{type(embeddings)}> is not currently handled.")

        separated_sentence_embeddings.append(separated_sentence_embedding)
        labels[i] = CLASSES[sentence.attributes["polarity"]]

    sentence_embeddings: NDArray = stack(separated_sentence_embeddings)
    return sentence_embeddings, labels


def get_best_estimator(embeddings: NDArray, labels: NDArray, grid: ParameterGrid) -> GaussianMixture:
    best_model: Optional[GaussianMixture] = None
    best_score: float = -inf
    print(f"Running <{len(grid)}> trials...")
    for (i, parameters) in enumerate(grid, start=1):
        print(f"Trial <{i}>: <{parameters}>.")
        new_estimator: GaussianMixture = GaussianMixture(**parameters)
        new_estimator.fit(embeddings)
        current_predictions: NDArray = new_estimator.predict(embeddings)
        current_score: float = f1_score(labels, current_predictions, average="macro")
        if current_score > best_score:
            print(f"Macro-F1 Score: {current_score} (current) > {best_score} (prior best)")
            best_model = new_estimator
            best_score = current_score
        else:
            print(f"Macro-F1 Score: {current_score} (current) < {best_score} (best)")

    return best_model


def load_unlabeled_embeddings(treebank_sentences: dict[str, list[ConllSentence]], embeddings: EmbeddingType,
                              lexicon: PolarityLexicon) -> list[NDArray]:
    treebank_sentence_embeddings: list[NDArray] = []
    for i, (treebank, sentences) in enumerate(treebank_sentences.items()):
        separated_sentence_embeddings: list[NDArray] = []
        for sentence in tqdm(sentences, desc="Unlabeled Sentences: "):
            lemmata: list[str] = []
            for j in range(0, len(sentence.tokens)):
                if MULTIWORD_ID_MARKER not in sentence.token_ids[j]:
                    lemmata.append(sentence.lemmas[j])

            if isinstance(embeddings, dict) is True:
                separated_sentence_embedding: NDArray = \
                    gather_aggregated_sentence_embedding(lemmata, embeddings, lexicon)
            elif isinstance(embeddings, SentenceTransformer) is True:
                separated_sentence_embedding: NDArray = \
                    create_sentence_embedding(sentence.sentence_text, embeddings, lemmata, lexicon)
            else:
                raise ValueError(f"The embedding of type <{type(embeddings)}> is not currently handled.")

            separated_sentence_embeddings.append(separated_sentence_embedding)
        else:
            individual_treebank_embeddings: NDArray = stack(separated_sentence_embeddings, axis=0)
            treebank_sentence_embeddings.append(individual_treebank_embeddings)

    return treebank_sentence_embeddings


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--components", type=int, default=len(list(PolarityClass)), help=GaussianMessage.COMPONENTS)
    parser.add_argument(
        "--embedding-filepath", type=str, default=EMBEDDING_PATH, help=GaussianMessage.EMBEDDING_FILEPATH
    )
    parser.add_argument("--input-filepath", type=str, default=TREEBANK_INPUT_PATH, help=AnnotatorMessage.INPUT_FILEPATH)
    parser.add_argument("--lexicon-filepath", type=str, default=LEXICON_PATH, help=AnnotatorMessage.LEXICON_FILEPATH)
    parser.add_argument("--output-filename", type=str, default=None, help=AnnotatorMessage.OUTPUT_FILENAME)
    parser.add_argument(
        "--output-filepath", type=str, default=TREEBANK_OUTPUT_PATH, help=AnnotatorMessage.OUTPUT_FILEPATH
    )
    parser.add_argument("--report", action=BooleanOptionalAction, default=False, help=AnnotatorMessage.REPORT)
    parser.add_argument("--random-seed", type=int, default=42, help=GeneralMessage.RANDOM_SEED)
    parser.add_argument("--random-seeds", type=int, default=10, help=GaussianMessage.RANDOM_SEEDS)
    parser.add_argument("--seed-filepath", type=str, default=SEED_FILEPATH, help=GaussianMessage.SEED_FILEPATH)
    args: Namespace = parser.parse_args()

    seed(args.random_seed)
    print(f"Starting Gaussian polarity annotator with random seed <{args.random_seed}> ...", flush=True)

    latin_lemmatizer: LatinBackoffLemmatizer = LatinBackoffLemmatizer()

    # First, we determine the best Gaussian model for clustering our data.
    if isdir(args.embedding_filepath) is True:
        print("Attempting to use SPhilBERTa embeddings...", flush=True)
        selected_embeddings: EmbeddingType = SentenceTransformer(args.embedding_filepath)
    elif isfile(args.embedding_filepath) is True:
        print("Attempting to use averaged word embeddings (from Burns *et al.* 2021)...", flush=True)
        selected_embeddings: EmbeddingType = load_embeddings(args.embedding_filepath)
    else:
        raise ValueError(f"The path <{args.embedding_filepath}> is not a valid file or directory.")

    polarity_lexicon: PolarityLexicon = get_polarity_lexicon(args.lexicon_filepath)
    labeled_dataset: PolarityDataset = PolarityDataset(args.seed_filepath)
    labeled_embeddings, polarity_labels = \
        load_labeled_embeddings(labeled_dataset, selected_embeddings, polarity_lexicon, latin_lemmatizer)

    COMPONENTS: tuple[int] = (args.components,)
    RANDOM_SEEDS: list[int] = [randrange(0, MAXIMUM_SIZE) for _ in range(0, args.random_seeds)]
    REGULARIZATION: tuple[float] = (1e-6 if isinstance(selected_embeddings, dict) else 1e-5,)

    parameter_grid: ParameterGrid = ParameterGrid({
        "covariance_type": COVARIANCE_TYPES,
        "init_params": INITIALIZATION_STRATEGIES,
        "n_init": NUMBER_INITIALIZATIONS,
        "n_components": COMPONENTS,
        "random_state": RANDOM_SEEDS,
        "max_iter": MAX_ITERATIONS,
        "reg_covar": REGULARIZATION
    })

    filterwarnings("ignore", category=UserWarning)
    gaussian_estimator: GaussianMixture = get_best_estimator(labeled_embeddings, polarity_labels, parameter_grid)
    print(f"Best Estimator: <{gaussian_estimator}>...", flush=True)

    print("Collecting input filepaths ...", flush=True)
    located_filepaths: list[tuple[str, str]] = collect_input_filepaths(args.input_filepath)

    print("Collecting output filepath ...", flush=True)
    output_kwargs: dict[str, Any] = {}
    if args.output_filename is not None:
        output_kwargs["output_filename"] = args.output_filename

    output_filepath: str = get_output_filepath(args.output_filepath, **output_kwargs)

    print("Gathering unlabeled sentences ...", flush=True)
    sentences_by_treebank: dict[str, list[ConllSentence]] = gather_treebank_sentences(located_filepaths)

    print("Gathering unlabeled sentence IDs ...", flush=True)
    sentence_ids: list[str] = []
    all_sentences: list[ConllSentence] = []
    for subdirectory, sentence_group in tqdm(sentences_by_treebank.items(), desc="Treebank Sentences: "):
        all_sentences.extend(sentence_group)
        sentence_group_ids: list[str] = construct_sentence_ids(sentence_group, subdirectory)
        sentence_ids.extend(sentence_group_ids)

    print("Loading unlabeled embeddings ...", flush=True)
    treebank_embeddings: list[NDArray] = \
        load_unlabeled_embeddings(sentences_by_treebank, selected_embeddings, polarity_lexicon)

    print("Gathering estimator classifications ...", flush=True)
    final_classifications: list[str] = []
    label_distributions: list[NDArray] = []
    for treebank_embedding in treebank_embeddings:
        treebank_size, *_ = treebank_embedding.shape
        chunks: list[NDArray] = split(treebank_embedding, list(range(0, treebank_size, EVALUATION_BATCH_SIZE)), axis=0)
        chunks: list[NDArray] = [chunk for chunk in chunks if chunk.shape[0] > 0]
        for chunk in tqdm(chunks, desc="Treebank Chunks: "):
            chunk_distribution: NDArray = gaussian_estimator.predict_proba(chunk)
            chunk_size, *_ = chunk_distribution.shape
            current_classifications: list[str] = \
                [INDEXED_CLASSES[argmax(chunk_distribution[i])] for i in range(0, chunk_size)]
            final_classifications.extend(current_classifications)
            label_distributions.append(chunk_distribution)

    print("Gathering distribution labels...", flush=True)
    concatenated_distributions: NDArray = concatenate(label_distributions, axis=0)
    listed_distributions: list[list[float]] = concatenated_distributions.tolist()
    labeled_distributions: list[dict[str, float]] = []
    for distribution in listed_distributions:
        positive, negative, neutral, mixed = distribution
        labeled_distribution: dict[str, float] = {
            PolarityClass.POSITIVE: positive,
            PolarityClass.NEGATIVE: negative,
            PolarityClass.NEUTRAL: neutral,
            PolarityClass.MIXED: mixed
        }
        labeled_distributions.append(labeled_distribution)

    if args.report is True:
        print("Reporting statistics...\n", flush=True)
        report_statistics(sentence_ids, all_sentences, final_classifications)

    print("Outputting results to file...", flush=True)
    output_polarity_tsv(output_filepath, sentence_ids, all_sentences, final_classifications, labeled_distributions)

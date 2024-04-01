from os import walk
from os.path import isdir, isfile
from statistics import mean, stdev
from typing import Optional

from cltk.lemmatize import LatinBackoffLemmatizer
from numpy import array, concatenate, int16, mean, stack, zeros
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import inference_mode

from utils.data.annotation import CLASSES, PolarityCoordinate, POINTS, PolarityClass
from utils.data.annotation.constants import DEFAULT_TREEBANK_NAME, EmbeddingTable, EmbeddingType
from utils.data.loaders.general.conll_loader import ConllSentence, ConllDataset, MULTIWORD_ID_MARKER
from utils.data.loaders.general.embedding_loader import load_embeddings
from utils.data.loaders.polarity import LexiconEntry, PolarityLexicon, construct_lexicon_mapping, load_lexicon_file, \
    PolarityDataset, PolaritySentence
from utils.data.tokenizers.tokens import WORD2VEC_UNK_TOKEN


def collect_input_filepaths(input_filepath: str) -> list[tuple[str, str]]:
    located_filepaths: list[tuple[str, str]] = []
    if isdir(input_filepath) is True:
        for item in walk(input_filepath):
            filepath, subdirectories, filenames = item
            current_subdirectory: str = filepath.split("\\")[-1]
            if len(filenames) > 0:
                filepath = filepath.replace('\\', '/')
                located_filepaths.extend([(f"{filepath}/{filename}", current_subdirectory) for filename in filenames])
    else:
        raise ValueError("Input filepath not to a valid directory.")

    return located_filepaths


def get_output_filepath(output_filepath: str, output_filename: str = DEFAULT_TREEBANK_NAME) -> str:
    if isdir(output_filepath):
        output_filepath = f"{output_filepath}/{output_filename}.tsv"
    else:
        raise ValueError(f"The output filepath, <{output_filepath}>, is not a valid directory.")

    return output_filepath


def gather_treebank_sentences(located_filepaths: list[tuple[str, str]]) -> dict[str, list[ConllSentence]]:
    treebank_sentences: dict[str, list[ConllSentence]] = {}
    for (filepath, subdirectory) in located_filepaths:
        if treebank_sentences.get(subdirectory, None) is None:
            treebank_sentences[subdirectory] = []
        treebank_sentences[subdirectory].extend(ConllDataset.load_conll_file(filepath))

    return treebank_sentences


def get_polarity_lexicon(lexicon_filepath: str) -> PolarityLexicon:
    if not isfile(lexicon_filepath):
        raise ValueError("Lexicon filepath not to a valid file.")
    else:
        lexicon_entries: list[LexiconEntry] = load_lexicon_file(lexicon_filepath)
        polarity_lexicon: PolarityLexicon = construct_lexicon_mapping(lexicon_entries)

    return polarity_lexicon


def construct_sentence_ids(sentences: list[ConllSentence], subdirectory_name: str) -> list[str]:
    ids: list[str] = []
    for sentence in sentences:
        sentence_id: str = f"{subdirectory_name.lower()}:{sentence.sentence_id}"
        ids.append(sentence_id)

    return ids


def output_polarity_tsv(output_filepath: str, ids: list[str], sentences: list[ConllSentence],
                        classifications: list[str], distributions: list[dict[str, float]]):
    assert len(ids) == len(sentences) == len(classifications) == len(distributions)
    with open(output_filepath, encoding="utf-8", mode="w+") as output_file:
        for i in range(0, len(ids)):
            output_distances: str = f"{distributions[i]['positive']};{distributions[i]['negative']};" \
                                    f"{distributions[i]['neutral']};{distributions[i]['mixed']}"
            output_line: str = f"{ids[i]}\t{sentences[i].sentence_text}\t{classifications[i]}\t{output_distances}\n"
            output_file.write(output_line)


def report_statistics(ids: list[str], sentences: list[ConllSentence], classifications: list[str],
                      lemmata: Optional[list[list[Optional[str]]]] = None):
    sentence_count: int = len(sentences)
    token_counts: list[int] = [
        len(sentence.tokens) - len(list(filter(lambda t: MULTIWORD_ID_MARKER in t, sentence.tokens)))
        for sentence in sentences   # We exclude multi-word tokens and only include their parts.
    ]
    total_token_count: int = sum(token_counts)
    average_token_count: int = round(mean(token_counts))
    total_token_deviation: int = round(stdev(token_counts))

    if lemmata is not None:
        unknown_token_counts: list[int] = [sentence_lemmata.count(None) for sentence_lemmata in lemmata]
        average_unknown_tokens: Optional[int] = round(mean(unknown_token_counts))
        unknown_tokens_deviation: Optional[int] = round(stdev(unknown_token_counts))
        unknown_token_count: Optional[int] = sum(unknown_token_counts)
    else:
        average_unknown_tokens = None
        unknown_tokens_deviation = None
        unknown_token_count = None

    sentence_by_treebank_count: dict[str, int] = {}
    for identifier in ids:
        dataset, *_ = identifier.split(":")
        if dataset not in sentence_by_treebank_count:
            sentence_by_treebank_count[dataset] = 0
        sentence_by_treebank_count[dataset] += 1

    sentence_by_classification: dict[str, int] = {}
    for classification in classifications:
        if classification not in sentence_by_classification:
            sentence_by_classification[classification] = 0
        sentence_by_classification[classification] += 1

    treebank_listing: str = "\n\t".join([f"{key}: {value}" for key, value in sentence_by_treebank_count.items()])
    classification_listing: str = "\n\t".join([f"{key}: {value}" for key, value in sentence_by_classification.items()])

    output_string: str = f"This dataset contains {sentence_count} sentences, totaling {total_token_count} tokens. " \
                         f"An average of {average_token_count} \u00B1 {total_token_deviation} " \
                         f"tokens were in each sentence."

    if lemmata is not None:
        output_string += f"\nA total of {total_token_count - unknown_token_count} tokens were known by the lexicon. " \
                         f"\nOn average, {average_unknown_tokens} \u00B1 {unknown_tokens_deviation} " \
                         f"were unknown per sentence."

    output_string += f"\nThe distribution of sentences by dataset is as follows:" \
                     f"\n\t{treebank_listing}" \
                     f"\n\nThe distribution of classes is as follows:" \
                     f"\n\t{classification_listing}"

    print(output_string)


def compute_polarity_coordinate(lemmata: list[str], lexicon: PolarityLexicon,
                                is_lexicon_sensitive: bool) -> PolarityCoordinate:
    polarities: list[float] = []
    intensities: list[float] = []
    for lemma in lemmata:
        if lemma is None:
            if is_lexicon_sensitive is True:
                continue
            else:
                polarities.append(0.5)
                intensities.append(0.5)
        else:
            polarities.append(lexicon[lemma]["score"])
            intensities.append(abs(lexicon[lemma]["score"]))

    assert len(polarities) == len(intensities)
    if len(polarities) > 0:
        polarity_score: float = (sum(polarities) / (2 * len(polarities))) + 0.5
        assert 0.0 <= polarity_score <= 1.0
        intensity_score: float = sum(intensities) / len(intensities)
        assert 0.0 <= intensity_score <= 1.0
        polarity_coordinate: PolarityCoordinate = PolarityCoordinate(polarity_score, intensity_score)
    else:
        # If we use the lexicon-sensitive flag, then it's possible no words are found,
        #   so we could divide by zero. This condition sets a default neutral polarity coordinate.
        polarity_coordinate: PolarityCoordinate = POINTS[PolarityClass.NEUTRAL]

    return polarity_coordinate


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


def gather_embeddings(embedding_filepath: str) -> EmbeddingType:
    # First, we determine the best Gaussian model for clustering our data.
    if isdir(embedding_filepath) is True:
        print("Attempting to use SPhilBERTa embeddings...", flush=True)
        selected_embeddings: EmbeddingType = SentenceTransformer(embedding_filepath)
    elif isfile(embedding_filepath) is True:
        print("Attempting to use averaged word embeddings (from Burns *et al.* 2021)...", flush=True)
        selected_embeddings: EmbeddingType = load_embeddings(embedding_filepath)
    else:
        raise ValueError(f"The path <{embedding_filepath}> is not a valid file or directory.")

    return selected_embeddings

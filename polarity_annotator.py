from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from math import dist, inf
from typing import Optional, Any

from cltk.lemmatize.lat import LatinBackoffLemmatizer

from utils.cli.messages import AnnotatorMessage, PCMessage
from utils.data.annotation import PolarityCoordinate, POINTS
from utils.data.annotation.helpers import collect_input_filepaths, get_output_filepath, gather_treebank_sentences, \
    get_polarity_lexicon, output_polarity_tsv, report_statistics, construct_sentence_ids, compute_polarity_coordinate
from utils.data.loaders.general.conll_loader import ConllSentence, MULTIWORD_ID_MARKER
from utils.data.loaders.polarity import PolarityLexicon

TREEBANK_INPUT_PATH: str = "data/polarity/training/unannotated"
LEXICON_PATH: str = "data/polarity/LatinAffectusv4.tsv"
TREEBANK_OUTPUT_PATH: str = "data/polarity/training/annotated/polarity-coordinate/full"


def gather_conll_lemmata(sentence: ConllSentence, lexicon: PolarityLexicon,
                         lemmatizer: LatinBackoffLemmatizer) -> list[Optional[str]]:
    lemmata: list[Optional[str]] = []
    backoff_lemmata: list[str] = [lemma for (word, lemma) in lemmatizer.lemmatize(sentence.tokens)]
    for i in range(0, len(sentence.tokens)):
        if MULTIWORD_ID_MARKER in sentence.token_ids[i]:   # We exclude multiword IDs.
            continue

        given_lemma: str = sentence.lemmas[i]
        if given_lemma in lexicon:
            lemmata.append(given_lemma)
        elif backoff_lemmata[i] in lexicon:
            lemmata.append(backoff_lemmata[i])
        else:
            lemmata.append(None)

    return lemmata


def classify_sentence(polarity_coordinate: PolarityCoordinate) -> tuple[list[str], dict[str, float]]:
    classifications: list[str] = []

    distances: dict[str, float] = {}
    for class_name, class_coordinate in POINTS.items():
        euclidean_distance: float = dist(polarity_coordinate, class_coordinate)
        distances[class_name] = euclidean_distance

    current_minimum: float = inf
    for (distance_name, distance) in distances.items():
        if distance < current_minimum:
            current_minimum = distance
            classifications.clear()
            classifications.append(distance_name)
        elif distance == current_minimum:
            classifications.append(distance_name)
        else:
            continue

    return classifications, distances


def break_classification_ties(classifications: list[list[str]]):
    sole_classifications: list[str] = []
    current_totals: dict[str, int] = {key: 0 for key in POINTS.keys()}
    for i in range(0, len(classifications)):
        if len(classifications[i]) == 1:
            sole_classification: str = classifications[i][-1]
            current_totals[sole_classification] += 1

    for i in range(0, len(classifications)):
        if len(classifications[i]) == 1:
            sole_classification: str = classifications[i][-1]
            sole_classifications.append(sole_classification)
        else:
            class_pairs: list[tuple[str, int]] = \
                [(classification, current_totals[classification]) for classification in classifications[i]]
            sole_classification, _ = min(class_pairs, key=lambda pair: pair[-1])
            sole_classifications.append(sole_classification)
            current_totals[sole_classification] += 1

    return sole_classifications


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--input-filepath", type=str, default=TREEBANK_INPUT_PATH, help=AnnotatorMessage.INPUT_FILEPATH)
    parser.add_argument("--lexicon-filepath", type=str, default=LEXICON_PATH, help=AnnotatorMessage.LEXICON_FILEPATH)
    parser.add_argument("--output-filename", type=str, default=None, help=AnnotatorMessage.OUTPUT_FILENAME)
    parser.add_argument(
        "--output-filepath", type=str, default=TREEBANK_OUTPUT_PATH, help=AnnotatorMessage.OUTPUT_FILEPATH
    )
    parser.add_argument(
        "--lexicon-sensitive", action=BooleanOptionalAction, default=False, help=PCMessage.LEXICON_SENSITIVE
    )
    parser.add_argument("--report", action=BooleanOptionalAction, default=False, help=AnnotatorMessage.REPORT)
    args: Namespace = parser.parse_args()

    located_filepaths: list[tuple[str, str]] = collect_input_filepaths(args.input_filepath)

    output_kwargs: dict[str, Any] = {}
    if args.output_filename is not None:
        output_kwargs["output_filename"] = args.output_filename

    output_path: str = get_output_filepath(args.output_filepath, **output_kwargs)
    treebank_sentences: dict[str, list[ConllSentence]] = gather_treebank_sentences(located_filepaths)
    polarity_lexicon: PolarityLexicon = get_polarity_lexicon(args.lexicon_filepath)

    sentence_ids: list[str] = []
    all_sentences: list[ConllSentence] = []
    for subdirectory, sentence_group in treebank_sentences.items():
        all_sentences.extend(sentence_group)
        sentence_group_ids: list[str] = construct_sentence_ids(sentence_group, subdirectory)
        sentence_ids.extend(sentence_group_ids)

    backoff_lemmatizer: LatinBackoffLemmatizer = LatinBackoffLemmatizer()
    all_lemmata: list[list[Optional[str]]] = []
    tentative_classifications: list[list[str]] = []
    polarity_distances: list[dict[str, float]] = []
    for current_sentence in all_sentences:
        current_lemmata: list[Optional[str]] = \
            gather_conll_lemmata(current_sentence, polarity_lexicon, backoff_lemmatizer)
        all_lemmata.append(current_lemmata)
        current_coordinate: PolarityCoordinate = \
            compute_polarity_coordinate(current_lemmata, polarity_lexicon, args.lexicon_sensitive)
        current_classifications, current_distances = classify_sentence(current_coordinate)
        tentative_classifications.append(current_classifications)
        polarity_distances.append(current_distances)

    final_classifications: list[str] = break_classification_ties(tentative_classifications)

    if args.report is True:
        report_statistics(sentence_ids, all_sentences, final_classifications, all_lemmata)

    output_polarity_tsv(output_path, sentence_ids, all_sentences, final_classifications, polarity_distances)

from argparse import ArgumentParser, Namespace
from csv import DictReader
from os import environ, listdir
from random import seed
from pickle import load
from platform import system
from typing import BinaryIO

from cltk.lemmatize import LatinBackoffLemmatizer
from numpy.typing import NDArray
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture

from utils.cli.messages import AnnotatorMessage, ComparerMessage, GaussianMessage, GeneralMessage
from utils.data.annotation.classes import INDEXED_CLASSES
from utils.data.annotation.constants import EMBEDDING_PATH, LEXICON_PATH, MODEL_FILEPATH, PREDICTIONS_FILEPATH, \
    TEST_FILEPATH, EmbeddingType
from utils.data.annotation.helpers import gather_embeddings, get_polarity_lexicon, load_labeled_embeddings
from utils.data.loaders.polarity import PolarityLexicon, PolarityDataset

# We conditionally set an environment variable.
if system() == "Windows":
    environ["OMP_NUM_THREADS"] = "1"


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--embedding-filepath", type=str, default=EMBEDDING_PATH, help=GaussianMessage.EMBEDDING_FILEPATH
    )
    parser.add_argument("--lexicon-filepath", type=str, default=LEXICON_PATH, help=AnnotatorMessage.LEXICON_FILEPATH)
    parser.add_argument("--model-filepath", type=str, default=MODEL_FILEPATH, help=ComparerMessage.MODEL_FILEPATH)
    parser.add_argument(
        "--predictions-directory", type=str, default=PREDICTIONS_FILEPATH, help=ComparerMessage.PREDICTIONS_DIRECTORY
    )
    parser.add_argument("--random-seed", type=int, default=42, help=GeneralMessage.RANDOM_SEED)
    parser.add_argument("--test-directory", type=str, default=TEST_FILEPATH, help=ComparerMessage.TEST_DIRECTORY)
    args: Namespace = parser.parse_args()

    seed(args.random_seed)
    print(f"Starting Gaussian results comparison with random seed <{args.random_seed}> ...", flush=True)

    latin_lemmatizer: LatinBackoffLemmatizer = LatinBackoffLemmatizer()
    selected_embeddings: EmbeddingType = gather_embeddings(args.embedding_filepath)
    polarity_lexicon: PolarityLexicon = get_polarity_lexicon(args.lexicon_filepath)
    model_file: BinaryIO = open(args.model_filepath, mode="rb")
    gaussian_estimator: GaussianMixture = load(model_file)

    # Load the test data used by the neural network.
    test_dataset: PolarityDataset = PolarityDataset(args.test_directory)
    labeled_test_embeddings, test_labels = \
        load_labeled_embeddings(test_dataset, selected_embeddings, polarity_lexicon, latin_lemmatizer)

    # Run the Gaussian model the test data and save the results:
    gaussian_predictions: NDArray = gaussian_estimator.predict(labeled_test_embeddings)
    test_score: float = f1_score(test_labels, gaussian_predictions, average="macro")
    print(f"Gaussian Macro-F1 Score: {test_score:.2f}")

    # Load the results from the neural network predictions:
    neural_labels: list[str] = []
    for filename in listdir(args.predictions_directory):
        with open(f"{args.predictions_directory}/{filename}", encoding="utf-8", mode="r") as test_file:
            results: DictReader = DictReader(test_file, dialect="excel-tab")
            for row in results:
                neural_labels.append(row["P"])

    gaussian_labels: list[str] = []
    for prediction in gaussian_predictions:
        gaussian_labels.append(INDEXED_CLASSES[prediction.item()])

    # Compare the two sets of results using agreement metric
    kappa = cohen_kappa_score(gaussian_labels, neural_labels)
    print(f"Cohen's Kappa Score for agreement: {kappa:.2f}\n")

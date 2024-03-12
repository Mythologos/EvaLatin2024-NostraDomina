from typing import Any, Optional

from torch import int64, Tensor, tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.constants import NamedEnum

from .lexicon_loader import LexiconEntry, LexiconEntryTable, PolarityLexicon, \
    construct_lexicon_mapping, load_lexicon_file
from .polarity_loader import PolarityDataset, PolaritySentence
from ...tokenizers import LatinLMTokenizer


class NamedPolarityDataset(NamedEnum):
    COORDINATE_TREEBANK: str = "coordinate-treebank"
    GAUSSIAN_TREEBANK: str = "gaussian-treebank"
    EVALATIN_2024_TEST: str = "evalatin-2024-test"
    HORACE_ODES: str = "horace-odes"


DATASET_LOCATIONS: dict[NamedPolarityDataset, dict[str, str]] = {
    NamedPolarityDataset.COORDINATE_TREEBANK: {
        "training": "data/polarity/training/annotated/polarity-coordinate/splits/training.tsv",
        "validation": "data/polarity/training/annotated/polarity-coordinate/splits/validation.tsv",
        "test": "data/polarity/training/annotated/polarity-coordinate/splits/test.tsv"
    },
    NamedPolarityDataset.EVALATIN_2024_TEST: {
        "test": "data/polarity/evalatin-test"
    },
    NamedPolarityDataset.GAUSSIAN_TREEBANK: {
        "training": "data/polarity/training/annotated/gaussian/splits/training.tsv",
        "validation": "data/polarity/training/annotated/gaussian/splits/validation.tsv",
        "test": "data/polarity/training/annotated/gaussian/splits/test.tsv"
    },
    NamedPolarityDataset.HORACE_ODES: {
        "test": "data/polarity/GoldStandardv1-Horace.tsv"
    }
}


def load_dataset(dataset_name: NamedPolarityDataset, mode: str, loading_kwargs: dict[str, Any]) -> \
        dict[str, DataLoader]:
    inference_split: str = loading_kwargs["inference_split"]
    polarity_splits: dict[str, DataLoader] = {}
    try:
        dataset_locations: dict[str, str] = DATASET_LOCATIONS[dataset_name]
        for split, location in dataset_locations.items():
            if mode == "train" and split == "training":
                training_dataset: PolarityDataset = PolarityDataset(location)
                training_sampler: RandomSampler = RandomSampler(training_dataset, generator=loading_kwargs["generator"])
                training_loader: DataLoader = \
                    DataLoader(training_dataset, sampler=training_sampler, **loading_kwargs["common"])
                polarity_splits["training"] = training_loader
            elif (mode == "train" and split == "validation") or \
                    (mode in ("evaluate", "predict") and split == inference_split):
                evaluation_dataset: PolarityDataset = PolarityDataset(location)
                evaluation_sampler: SequentialSampler = SequentialSampler(evaluation_dataset)
                evaluation_loader: DataLoader = \
                    DataLoader(evaluation_dataset, sampler=evaluation_sampler, **loading_kwargs["common"])
                polarity_splits["evaluation"] = evaluation_loader
            else:
                continue
    except KeyError:
        raise ValueError(f"The dataset <{dataset_name}> is not currently supported.")

    if len(polarity_splits) == 0:
        raise ValueError(f"The designated dataset <{dataset_name}> and mode <{mode}> combination is not supported.")

    return polarity_splits


class LatinLMCollator:
    def __init__(self, tokenizer: LatinLMTokenizer, vocabulary: dict[str, int], maximum_length: int = 512,
                 should_pretokenize: bool = True, distance_type: Optional[str] = None):
        self.maximum_length = maximum_length
        self.tokenizer = tokenizer
        self.vocabulary: dict[str, int] = vocabulary

        self.distance_type = distance_type
        self.should_pretokenize = should_pretokenize

    def __call__(self, sentences: list[PolaritySentence]) -> \
            tuple[Tensor, list[Optional[str]], list[float], dict[str, Any]]:
        if self.should_pretokenize is True:
            pretokenized_sentences: list[list[str]] = [sentence.sentence_text.split() for sentence in sentences]
        else:
            pretokenized_sentences: list[list[str]] = [[sentence.sentence_text] for sentence in sentences]

        batch, batch_kwargs = self.prepare_batch(pretokenized_sentences)
        batch_kwargs["sentences"] = sentences

        polarities: list[Optional[str]] = [sentence.attributes.get("polarity", None) for sentence in sentences]

        distances: list[float] = []
        if self.distance_type is not None:
            for sentence in sentences:
                raw_distances: dict[str, float] = sentence.attributes["distances"]
                normalized_distances: list[float] = \
                    [distance / sum(raw_distances.values()) for distance in raw_distances.values()]
                distances.append(1.0 - min(normalized_distances))

        return batch, polarities, distances, batch_kwargs

    def set_collator_options(self, should_pretokenize: bool, distance_type: Optional[str] = None):
        self.should_pretokenize = should_pretokenize
        self.distance_type = distance_type

    def prepare_batch(self, batch: list[list[str]]) -> tuple[Tensor, dict[str, Any]]:
        batch_kwargs: dict[str, Any] = {}
        subword_indices: list[list[int]] = [
            [
                index for word in sentence
                for index in self.tokenizer.subword_tokenizer.encode(word, **self.tokenizer.encoding_kwargs)
            ]
            for sentence in batch
        ]

        class_index: int = self.vocabulary[self.tokenizer.special_tokens.CLASS]
        separation_index: int = self.vocabulary[self.tokenizer.special_tokens.SEPARATION]
        for subword_chunk in subword_indices:
            subword_chunk.insert(0, class_index)
            subword_chunk.append(separation_index)

        sequence_lengths: list[int] = [len(sequence) for sequence in subword_indices]
        max_sequence_length: int = min(max(sequence_lengths), self.maximum_length)
        padding_index: int = self.vocabulary[self.tokenizer.special_tokens.PADDING]
        for chunk_index, subword_chunk in enumerate(subword_indices):
            if len(subword_chunk) > max_sequence_length:
                for _ in range(0, len(subword_chunk) - max_sequence_length):
                    subword_chunk.pop(-2)   # We use -2 to avoid popping the [SEP] token.
                else:
                    sequence_lengths[chunk_index] = max_sequence_length
            else:
                for _ in range(0, max_sequence_length - len(subword_chunk)):
                    subword_chunk.append(padding_index)

        index_tensor: Tensor = tensor(subword_indices, dtype=int64)
        embedding_padding_mask: Tensor = (index_tensor != padding_index)   # type: ignore
        encoder_padding_mask: Tensor = (index_tensor == padding_index)   # type: ignore
        batch_kwargs["embedding"] = {"attention_mask": embedding_padding_mask}
        batch_kwargs["encoder"] = {"padding_mask": encoder_padding_mask, "sequence_lengths": sequence_lengths}

        return index_tensor, batch_kwargs

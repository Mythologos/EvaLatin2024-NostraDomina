from os import listdir
from os.path import isfile, isdir
from typing import Any, NamedTuple

from torch.utils.data import Dataset

from utils.data.annotation import PolarityClass


class PolaritySentence(NamedTuple):
    sentence_id: str
    sentence_text: str
    attributes: dict[str, Any] = {}


class PolarityDataset(Dataset):
    def __init__(self, split_filepath: str):
        filepaths: list[str] = self._collect_filepaths(split_filepath)
        self.data: list[PolaritySentence] = self.load_polarity_file(filepaths)

    @staticmethod
    def _collect_filepaths(main_filepath: str) -> list[str]:
        filepaths: list[str] = []
        if isfile(main_filepath) is True:
            filepaths.append(main_filepath)
        elif isdir(main_filepath) is True:
            filenames: list[str] = listdir(main_filepath)
            for filename in filenames:
                if filename.endswith(".tsv") is True:
                    filepaths.append(f"{main_filepath}/{filename}")
        else:
            raise ValueError(f"The path <{main_filepath}> is not a valid file or directory.")

        return filepaths

    @staticmethod
    def load_polarity_file(input_filepaths: list[str]) -> list[PolaritySentence]:
        polarity_sentences: list[PolaritySentence] = []
        for filepath in input_filepaths:
            with open(filepath, encoding="utf-8", mode="r") as polarity_file:
                for line_index, line in enumerate(polarity_file):
                    sentence_id, sentence, *other = line.strip().split("\t")

                    attributes: dict[str, Any] = {"source": filepath, "line": line_index}
                    if len(other) >= 1:
                        attributes["polarity"] = other.pop(0)

                    if len(other) >= 1:
                        positive, negative, neutral, mixed = other[-1].split(";")
                        attributes["distances"] = {
                            PolarityClass.POSITIVE: float(positive),
                            PolarityClass.NEGATIVE: float(negative),
                            PolarityClass.NEUTRAL: float(neutral),
                            PolarityClass.MIXED: float(mixed)
                        }

                    polarity_sentence: PolaritySentence = PolaritySentence(sentence_id, sentence, attributes)
                    polarity_sentences.append(polarity_sentence)

        return polarity_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

from re import Match, fullmatch
from typing import Any, NamedTuple, Union

from numpy import int16, zeros
from numpy.typing import NDArray
from torch.utils.data import Dataset

ARCHIMEDES_NEWPAR_LINE: str = "# newpar"
METADATA_PARSER: str = r"# (?P<heading>sent_id|text|reference|newdoc id|source|citation_hierarchy|newpar)" \
                       r"[\s]=[\s](?P<content>.+)"
MULTIWORD_ID_MARKER: str = "-"


class ConllSentence(NamedTuple):
    sentence_id: str
    sentence_text: str
    sentence_metadata: dict[str, Any]
    token_ids: list[str]
    tokens: list[str]
    lemmas: list[str]
    universal_pos_tags: list[str]
    exclusive_pos_tags: list[str]
    features: list[str]
    heads: list[str]
    dependency_relations: list[str]
    enhanced_relations: list[str]
    miscellany: list[str]


ConllTokens = list[str]
AdjacencyMatrix = NDArray[int]
LabelList = list[str]
ParsingElement = tuple[ConllTokens, AdjacencyMatrix, LabelList, ConllSentence]


class ConllDataset(Dataset):
    def __init__(self, filepaths: list[str]):
        self.data: list[ParsingElement] = self.load_parsing_data(filepaths)

    def load_parsing_data(self, input_filepaths: list[str]) -> list[ParsingElement]:
        conll_sentences: list[ConllSentence] = []

        for input_filepath in input_filepaths:
            current_conll_sentences: list[ConllSentence] = self.load_conll_file(input_filepath)
            conll_sentences.extend(current_conll_sentences)

        elements: list[ParsingElement] = [self.load_parsing_element(sentence) for sentence in conll_sentences]
        return elements

    @staticmethod
    def load_parsing_element(conll_sentence: ConllSentence) -> ParsingElement:
        tokens: list[str] = []
        arc_labels: LabelList = []
        adjacency_mapping: dict[int, int] = {}

        multiword_count: int = 0
        for i in range(0, len(conll_sentence.tokens)):
            if MULTIWORD_ID_MARKER in conll_sentence.token_ids[i]:
                multiword_count += 1
                continue
            else:
                tokens.append(conll_sentence.tokens[i])
                main_label, *_ = conll_sentence.dependency_relations[i].split(":")
                arc_labels.append(main_label)
                adjacency_mapping[i - multiword_count + 1] = int(conll_sentence.heads[i])

        # We need to account for the root node, hence the addition of a row and column.
        adjacency_matrix: AdjacencyMatrix = zeros((len(tokens) + 1, len(tokens) + 1), dtype=int16)
        for child_index, parent_index in adjacency_mapping.items():
            adjacency_matrix[parent_index, child_index] = 1

        return tokens, adjacency_matrix, arc_labels, conll_sentence

    @staticmethod
    def load_conll_file(input_filepath: str) -> list[ConllSentence]:
        conll_sentences: list[ConllSentence] = []
        with open(input_filepath, encoding="utf-8", mode="r") as conll_file:
            sentence_lines: list[str] = []
            for line in conll_file:
                line = line.strip()
                if line != "":
                    sentence_lines.append(line)
                else:
                    conll_sentence: ConllSentence = \
                        ConllDataset._parse_sentence(sentence_lines, len(conll_sentences) + 1)
                    conll_sentences.append(conll_sentence)
                    sentence_lines.clear()

        return conll_sentences

    @staticmethod
    def _parse_sentence(sentence_lines: list[str], sentence_number: int) -> ConllSentence:
        conll_metadata_attributes: dict[str, Union[str, dict[str, Any]]] = {}
        conll_sentence_attributes: dict[str, Union[str, list[str], dict[str, Any]]] = {
            "token_ids": [],
            "tokens": [],
            "lemmas": [],
            "universal_pos_tags": [],
            "exclusive_pos_tags": [],
            "features": [],
            "heads": [],
            "dependency_relations": [],
            "enhanced_relations": [],
            "miscellany": []
        }

        sentence_metadata: dict[str, Any] = {}
        for line in sentence_lines:
            if line.startswith("#") is True:
                if line == ARCHIMEDES_NEWPAR_LINE:
                    sentence_metadata["newpar"] = True
                else:
                    metadata_element: Match = fullmatch(METADATA_PARSER, line)
                    match metadata_element["heading"]:
                        case "sent_id":
                            conll_metadata_attributes["sentence_id"] = metadata_element["content"]
                        case "text":
                            conll_metadata_attributes["sentence_text"] = metadata_element["content"]
                        case _:
                            sentence_metadata[metadata_element["heading"]] = metadata_element["content"]
            else:
                conll_columns: list[str] = line.split("\t")
                for index, value in enumerate(conll_sentence_attributes.values()):
                    value.append(conll_columns[index])
        else:
            conll_metadata_attributes["sentence_metadata"] = sentence_metadata

        # If the data (like the Archimedes Latinus training data) does not include sentence IDs and text,
        #   we provide them.
        if conll_metadata_attributes.get("sentence_id", None) is None:
            conll_metadata_attributes["sentence_id"] = str(sentence_number)

        if conll_metadata_attributes.get("sentence_text", None) is None:
            conll_metadata_attributes["sentence_text"] = " ".join(conll_sentence_attributes["tokens"])

        conll_sentence_attributes.update(conll_metadata_attributes)
        conll_sentence: ConllSentence = ConllSentence(**conll_sentence_attributes)
        return conll_sentence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

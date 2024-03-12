from typing import NamedTuple, Union


LexiconEntryTable = dict[str, Union[float, int, str]]
PolarityLexicon = dict[str, LexiconEntryTable]


class LexiconEntry(NamedTuple):
    entry_id: int
    lemma: str
    part_of_speech: str
    polarity_score: float
    polarity_class: str
    provenance: str


def load_lexicon_file(input_filepath: str) -> list[LexiconEntry]:
    lexicon_entries: list[LexiconEntry] = []
    with open(input_filepath, encoding="utf-8", mode="r") as lexicon_file:
        for line_number, line in enumerate(lexicon_file, start=0):
            if line_number == 0:
                continue
            else:
                lemma, pos, score, classification, provenance = line.strip().split("\t")
                score: float = float(score)
                lexicon_entry: LexiconEntry = LexiconEntry(line_number, lemma, pos, score, classification, provenance)
                lexicon_entries.append(lexicon_entry)

    return lexicon_entries


def construct_lexicon_mapping(lexicon_entries: list[LexiconEntry]) -> PolarityLexicon:
    lexicon: PolarityLexicon = {}
    for entry in lexicon_entries:
        entry_data: LexiconEntryTable = {
            "id": entry.entry_id,
            "pos": entry.part_of_speech,
            "score": entry.polarity_score,
            "class": entry.polarity_class,
            "provenance": entry.provenance
        }
        lexicon[entry.lemma] = entry_data
    return lexicon

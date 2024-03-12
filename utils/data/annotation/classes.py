from utils.constants import NamedEnum


class PolarityClass(NamedEnum):
    POSITIVE: str = "positive"
    NEGATIVE: str = "negative"
    NEUTRAL: str = "neutral"
    MIXED: str = "mixed"


CLASSES: dict[str, int] = {
    PolarityClass.POSITIVE.value: 0,
    PolarityClass.NEGATIVE.value: 1,
    PolarityClass.NEUTRAL.value: 2,
    PolarityClass.MIXED.value: 3
}

INDEXED_CLASSES: dict[int, str] = {value: key for key, value in CLASSES.items()}

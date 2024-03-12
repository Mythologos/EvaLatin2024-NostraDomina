from typing import NamedTuple

from utils.data.annotation.classes import PolarityClass


class PolarityCoordinate(NamedTuple):
    polarity: float
    intensity: float


POINTS: dict[PolarityClass, PolarityCoordinate] = {
    PolarityClass.POSITIVE: PolarityCoordinate(1.0, 0.5),
    PolarityClass.NEGATIVE: PolarityCoordinate(0.0, 0.5),
    PolarityClass.NEUTRAL: PolarityCoordinate(0.5, 0.0),
    PolarityClass.MIXED: PolarityCoordinate(0.5, 1.0)
}

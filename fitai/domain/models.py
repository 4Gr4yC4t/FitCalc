from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FeatureVector:
    """Numeric features prepared for a single model inference."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.values)
        if not values:
            raise ValueError("FeatureVector must contain at least one value")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("FeatureVector values must all be finite numbers")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class InferenceResult:
    """Flat numeric output returned by the model."""

    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        scores = tuple(float(score) for score in self.scores)
        if not scores:
            raise ValueError("InferenceResult must contain at least one score")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("InferenceResult scores must all be finite numbers")
        object.__setattr__(self, "scores", scores)

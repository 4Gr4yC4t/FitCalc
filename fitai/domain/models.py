from __future__ import annotations

from dataclasses import dataclass, field
import math
from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4


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
    metadata: InferenceMetadata | None = None

    def __post_init__(self) -> None:
        scores = tuple(float(score) for score in self.scores)
        if not scores:
            raise ValueError("InferenceResult must contain at least one score")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("InferenceResult scores must all be finite numbers")
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    age: int
    height_cm: float
    weight_kg: float
    experience_level: str


@dataclass(frozen=True)
class UserGoal:
    goal_type: str
    target_date: date | None = None


@dataclass(frozen=True)
class TrainingHistory:
    workouts: tuple[Any, ...] = ()


@dataclass(frozen=True)
class NutritionLog:
    entries: tuple[Any, ...] = ()


@dataclass(frozen=True)
class BodyWeightHistory:
    measurements: tuple[Any, ...] = ()


@dataclass(frozen=True)
class FitnessUserContext:
    user_id: str
    profile: UserProfile
    goal: UserGoal
    training_history: TrainingHistory
    nutrition_log: NutritionLog
    body_weight_history: BodyWeightHistory


@dataclass(frozen=True)
class WorkoutPlan:
    exercises: tuple[Any, ...] = ()
    intensity: float = 0.0


@dataclass(frozen=True)
class NutritionRecommendation:
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float


@dataclass(frozen=True)
class OvertrainingAlert:
    is_overtrained: bool
    message: str


@dataclass(frozen=True)
class ProModeSuggestion:
    suggested: bool
    reason: str


@dataclass(frozen=True)
class RecommendationExplanation:
    text: str
    key_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceMetadata:
    model_version: str
    calculated_at: datetime


@dataclass(frozen=True)
class FitnessRecommendation:
    workout_plan: WorkoutPlan
    nutrition: NutritionRecommendation
    overtraining_alert: OvertrainingAlert
    pro_mode_suggestion: ProModeSuggestion
    explanation: RecommendationExplanation
    metadata: InferenceMetadata | None = None


@dataclass(frozen=True)
class AiAgent:
    name: str
    version: str
    strategy: str
    parameters: dict[str, str] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class PerformanceMetric:
    agent_id: UUID
    timestamp: datetime
    inference_ms: float
    current_fps: float
    gc_allocated_mb: float = 0.0
    false_positives: int = 0


@dataclass(frozen=True)
class PlayerProgress:
    player_name: str
    unlocked_difficulty: str
    inventory: dict[str, int]
    id: int | None = None

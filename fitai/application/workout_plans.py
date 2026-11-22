from enum import Enum

from fitai.application.ports import IWorkoutDecisionMaker
from fitai.domain.models import (
    BodyWeightHistory,
    FitnessUserContext,
    NutritionLog,
    TrainingHistory,
    UserGoal,
    UserProfile,
    WorkoutPlan,
)


class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ExperiencePlanBuilder:
    """Build a general workout from the selected experience level."""

    _PLANS = {
        ExperienceLevel.BEGINNER: WorkoutPlan(
            ("Ходьба — 10 минут", "Приседания к стулу — 2 × 8",
             "Отжимания от стены — 2 × 8"), 0.35,
        ),
        ExperienceLevel.INTERMEDIATE: WorkoutPlan(
            ("Ходьба — 5 минут", "Приседания — 3 × 10",
             "Отжимания от опоры — 3 × 10", "Планка — 3 × 20 секунд"), 0.55,
        ),
        ExperienceLevel.ADVANCED: WorkoutPlan(
            ("Динамическая разминка — 8 минут", "Выпады — 4 × 10 на каждую ногу",
             "Отжимания — 4 × 10", "Боковая планка — 3 × 30 секунд на сторону"), 0.75,
        ),
    }

    def build(self, context: FitnessUserContext) -> WorkoutPlan:
        return self._PLANS[ExperienceLevel(context.profile.experience_level)]


class WorkoutPlanService:
    def __init__(self, decision_maker: IWorkoutDecisionMaker) -> None:
        self._decision_maker = decision_maker

    def plan_for_level(self, level: ExperienceLevel) -> WorkoutPlan:
        # The level-only flow has no user account or personal data.
        # The default builder reads only experience_level from this sample context.
        context = FitnessUserContext(
            user_id="demo",
            profile=UserProfile("demo", age=30, height_cm=170, weight_kg=70,
                                experience_level=level.value),
            goal=UserGoal("general_fitness"),
            training_history=TrainingHistory(),
            nutrition_log=NutritionLog(),
            body_weight_history=BodyWeightHistory(),
        )
        return self._decision_maker.create_plan(context, None)

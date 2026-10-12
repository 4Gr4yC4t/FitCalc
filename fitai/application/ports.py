from typing import Protocol, runtime_checkable

from fitai.domain.models import (
    BodyWeightHistory,
    FeatureVector,
    FitnessRecommendation,
    FitnessUserContext,
    InferenceResult,
    NutritionLog,
    NutritionRecommendation,
    OvertrainingAlert,
    RecommendationExplanation,
    TrainingHistory,
    UserGoal,
    UserProfile,
    WorkoutPlan,
)


class InferenceUnavailableError(RuntimeError):
    """Raised when inference cannot produce a result from the configured model."""


class IAIController(Protocol):
    def build_recommendation(self, user_id: str) -> FitnessRecommendation: ...


@runtime_checkable
class IProfileProvider(Protocol):
    def get_profile(self, user_id: str) -> UserProfile: ...

    def get_goal(self, user_id: str) -> UserGoal: ...


@runtime_checkable
class ITrainingHistoryProvider(Protocol):
    def get_training_history(self, user_id: str) -> TrainingHistory: ...


@runtime_checkable
class INutritionDataProvider(Protocol):
    def get_nutrition_log(self, user_id: str) -> NutritionLog: ...

    def get_weight_history(self, user_id: str) -> BodyWeightHistory: ...


class IFeatureBuilder(Protocol):
    def build(self, context: FitnessUserContext) -> FeatureVector: ...


class IInferenceService(Protocol):
    def run(self, features: FeatureVector) -> InferenceResult: ...


class IWorkoutDecisionMaker(Protocol):
    def create_plan(
        self,
        context: FitnessUserContext,
        result: InferenceResult | None,
    ) -> WorkoutPlan: ...


class IRuleBasedFallback(IWorkoutDecisionMaker, Protocol):
    pass


class IWorkoutPlanGenerator(Protocol):
    def generate(
        self,
        context: FitnessUserContext,
        result: InferenceResult,
    ) -> WorkoutPlan: ...


class IRuleBasedPlanBuilder(Protocol):
    def build(self, context: FitnessUserContext) -> WorkoutPlan: ...


class INutritionRecommendationService(Protocol):
    def recommend(self, context: FitnessUserContext) -> NutritionRecommendation: ...


class IOvertrainingDetector(Protocol):
    def detect(self, context: FitnessUserContext) -> OvertrainingAlert: ...


class IRecommendationFormatter(Protocol):
    def explain(
        self,
        recommendation: FitnessRecommendation,
    ) -> RecommendationExplanation: ...


class IDataQualityValidator(Protocol):
    def validate(self, context: FitnessUserContext) -> bool: ...


class IFitCalcApiClient(Protocol):
    def get_profile(self, user_id: str) -> UserProfile: ...

    def get_goal(self, user_id: str) -> UserGoal: ...

    def get_training_history(self, user_id: str) -> TrainingHistory: ...

    def get_nutrition_log(self, user_id: str) -> NutritionLog: ...

    def get_weight_history(self, user_id: str) -> BodyWeightHistory: ...

    def post_recommendation(
        self,
        user_id: str,
        recommendation: FitnessRecommendation,
    ) -> None: ...

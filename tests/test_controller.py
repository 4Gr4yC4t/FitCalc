import unittest
from datetime import datetime, timezone

from fitai.application.controller import FitAIController
from fitai.application.ports import InferenceUnavailableError
from fitai.domain.models import (
    BodyWeightHistory,
    FeatureVector,
    InferenceMetadata,
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


class FakeProfileProvider:
    def get_profile(self, user_id: str) -> UserProfile:
        return UserProfile(user_id, 30, 180, 80, "beginner")

    def get_goal(self, user_id: str) -> UserGoal:
        return UserGoal("strength")


class FakeTrainingHistoryProvider:
    def get_training_history(self, user_id: str) -> TrainingHistory:
        return TrainingHistory()


class FakeNutritionDataProvider:
    def get_nutrition_log(self, user_id: str) -> NutritionLog:
        return NutritionLog()

    def get_weight_history(self, user_id: str) -> BodyWeightHistory:
        return BodyWeightHistory()


class FakeFeatureBuilder:
    def __init__(self) -> None:
        self.called = False

    def build(self, context) -> FeatureVector:
        self.called = True
        return FeatureVector((1.0, 2.0))


class FakeInferenceService:
    def __init__(self, result=None, error=None) -> None:
        self.result = result or InferenceResult(
            (0.7, 0.3),
            InferenceMetadata("test-model", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        )
        self.error = error
        self.called = False

    def run(self, features: FeatureVector) -> InferenceResult:
        self.called = True
        if self.error is not None:
            raise self.error
        return self.result


class FakeDecisionMaker:
    def __init__(self) -> None:
        self.results = []

    def create_plan(self, context, result) -> WorkoutPlan:
        self.results.append(result)
        return WorkoutPlan(intensity=0.7 if result is not None else 0.3)


class FakeNutritionService:
    def recommend(self, context) -> NutritionRecommendation:
        return NutritionRecommendation(2200, 150, 70, 240)


class FakeOvertrainingDetector:
    def detect(self, context) -> OvertrainingAlert:
        return OvertrainingAlert(False, "No overload detected")


class FakeFormatter:
    def explain(self, recommendation) -> RecommendationExplanation:
        return RecommendationExplanation("Based on the user's training history")


class FakeDataQualityValidator:
    def __init__(self, valid: bool = True) -> None:
        self.valid = valid

    def validate(self, context) -> bool:
        return self.valid


class FakeApiClient:
    def __init__(self) -> None:
        self.published = []

    def post_recommendation(self, user_id, recommendation) -> None:
        self.published.append((user_id, recommendation))


class FitAIControllerTests(unittest.TestCase):
    def create_controller(self, *, valid_data=True, inference_error=None):
        self.feature_builder = FakeFeatureBuilder()
        self.inference_service = FakeInferenceService(error=inference_error)
        self.decision_maker = FakeDecisionMaker()
        self.api_client = FakeApiClient()
        controller = FitAIController(
            profile_provider=FakeProfileProvider(),
            training_history_provider=FakeTrainingHistoryProvider(),
            nutrition_data_provider=FakeNutritionDataProvider(),
            feature_builder=self.feature_builder,
            inference_service=self.inference_service,
            decision_maker=self.decision_maker,
            nutrition_service=FakeNutritionService(),
            overtraining_detector=FakeOvertrainingDetector(),
            formatter=FakeFormatter(),
            data_quality_validator=FakeDataQualityValidator(valid_data),
            api_client=self.api_client,
        )
        return controller

    def test_uses_injected_services_to_build_and_publish_recommendation(self) -> None:
        controller = self.create_controller()

        recommendation = controller.build_recommendation("user-123")

        self.assertTrue(self.feature_builder.called)
        self.assertEqual(len(self.decision_maker.results), 1)
        self.assertIsNotNone(self.decision_maker.results[0])
        self.assertEqual(recommendation.metadata.model_version, "test-model")
        self.assertEqual(
            recommendation.explanation.text,
            "Based on the user's training history",
        )
        self.assertEqual(self.api_client.published, [("user-123", recommendation)])

    def test_uses_fallback_when_inference_is_unavailable(self) -> None:
        controller = self.create_controller(
            inference_error=InferenceUnavailableError("model is unavailable")
        )

        recommendation = controller.build_recommendation("user-123")

        self.assertTrue(self.feature_builder.called)
        self.assertEqual(self.decision_maker.results, [None])
        self.assertEqual(recommendation.workout_plan.intensity, 0.3)
        self.assertEqual(recommendation.metadata.model_version, "fallback")
        self.assertEqual(len(self.api_client.published), 1)

    def test_uses_fallback_without_inference_when_data_is_incomplete(self) -> None:
        controller = self.create_controller(valid_data=False)

        controller.build_recommendation("user-123")

        self.assertFalse(self.feature_builder.called)
        self.assertFalse(self.inference_service.called)
        self.assertEqual(self.decision_maker.results, [None])

    def test_unexpected_inference_errors_are_not_silently_replaced(self) -> None:
        controller = self.create_controller(inference_error=ValueError("bad model"))

        with self.assertRaisesRegex(ValueError, "bad model"):
            controller.build_recommendation("user-123")

        self.assertEqual(self.decision_maker.results, [])
        self.assertEqual(self.api_client.published, [])


if __name__ == "__main__":
    unittest.main()

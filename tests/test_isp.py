import unittest

from fitai.application.ports import (
    INutritionDataProvider,
    IProfileProvider,
    ITrainingHistoryProvider,
)
from fitai.domain.models import BodyWeightHistory, NutritionLog, TrainingHistory, UserGoal, UserProfile


class ProfileOnlyProvider:
    def get_profile(self, user_id: str) -> UserProfile:
        return UserProfile(user_id, 30, 180, 80, "beginner")

    def get_goal(self, user_id: str) -> UserGoal:
        return UserGoal("strength")


class TrainingOnlyProvider:
    def get_training_history(self, user_id: str) -> TrainingHistory:
        return TrainingHistory()


class NutritionOnlyProvider:
    def get_nutrition_log(self, user_id: str) -> NutritionLog:
        return NutritionLog()

    def get_weight_history(self, user_id: str) -> BodyWeightHistory:
        return BodyWeightHistory()


class InterfaceSegregationTests(unittest.TestCase):
    def test_each_focused_provider_satisfies_only_its_contract(self) -> None:
        profile_provider = ProfileOnlyProvider()
        training_provider = TrainingOnlyProvider()
        nutrition_provider = NutritionOnlyProvider()

        self.assertIsInstance(profile_provider, IProfileProvider)
        self.assertNotIsInstance(profile_provider, ITrainingHistoryProvider)
        self.assertNotIsInstance(profile_provider, INutritionDataProvider)

        self.assertIsInstance(training_provider, ITrainingHistoryProvider)
        self.assertNotIsInstance(training_provider, IProfileProvider)
        self.assertNotIsInstance(training_provider, INutritionDataProvider)

        self.assertIsInstance(nutrition_provider, INutritionDataProvider)
        self.assertNotIsInstance(nutrition_provider, IProfileProvider)
        self.assertNotIsInstance(nutrition_provider, ITrainingHistoryProvider)


if __name__ == "__main__":
    unittest.main()

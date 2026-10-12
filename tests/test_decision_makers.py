import unittest

from fitai.application.decision_makers import (
    NeuralNetworkDecisionMaker,
    RuleBasedFallback,
)
from fitai.domain.models import (
    BodyWeightHistory,
    FitnessUserContext,
    InferenceResult,
    NutritionLog,
    TrainingHistory,
    UserGoal,
    UserProfile,
    WorkoutPlan,
)


class FakeWorkoutPlanGenerator:
    def __init__(self) -> None:
        self.received_result = None

    def generate(self, context, result) -> WorkoutPlan:
        self.received_result = result
        return WorkoutPlan(intensity=0.8)


class FakeRuleBasedPlanBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, context) -> WorkoutPlan:
        self.calls += 1
        return WorkoutPlan(intensity=0.3)


def make_context() -> FitnessUserContext:
    return FitnessUserContext(
        user_id="user-123",
        profile=UserProfile("user-123", 30, 180, 80, "beginner"),
        goal=UserGoal("strength"),
        training_history=TrainingHistory(),
        nutrition_log=NutritionLog(),
        body_weight_history=BodyWeightHistory(),
    )


class DecisionMakerContractTests(unittest.TestCase):
    def test_neural_strategy_uses_model_plan_generator_when_result_exists(self) -> None:
        plan_generator = FakeWorkoutPlanGenerator()
        rule_builder = FakeRuleBasedPlanBuilder()
        strategy = NeuralNetworkDecisionMaker(plan_generator, rule_builder)
        inference_result = InferenceResult((0.8, 0.2))

        plan = strategy.create_plan(make_context(), inference_result)

        self.assertEqual(plan.intensity, 0.8)
        self.assertIs(plan_generator.received_result, inference_result)
        self.assertEqual(rule_builder.calls, 0)

    def test_neural_strategy_uses_rules_when_model_result_is_missing(self) -> None:
        rule_builder = FakeRuleBasedPlanBuilder()
        strategy = NeuralNetworkDecisionMaker(FakeWorkoutPlanGenerator(), rule_builder)

        plan = strategy.create_plan(make_context(), None)

        self.assertEqual(plan.intensity, 0.3)
        self.assertEqual(rule_builder.calls, 1)

    def test_both_strategies_share_contract_for_same_inputs(self) -> None:
        context = make_context()
        rule_builder = FakeRuleBasedPlanBuilder()
        strategies = (
            NeuralNetworkDecisionMaker(FakeWorkoutPlanGenerator(), rule_builder),
            RuleBasedFallback(rule_builder),
        )

        for result in (None, InferenceResult((0.6, 0.4))):
            for strategy in strategies:
                with self.subTest(
                    strategy=type(strategy).__name__,
                    has_inference=result is not None,
                ):
                    plan = strategy.create_plan(context, result)
                    self.assertIsInstance(plan, WorkoutPlan)
                    self.assertGreaterEqual(plan.intensity, 0.0)
                    self.assertLessEqual(plan.intensity, 1.0)


if __name__ == "__main__":
    unittest.main()

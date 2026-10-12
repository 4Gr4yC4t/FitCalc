from fitai.application.ports import (
    IRuleBasedPlanBuilder,
    IWorkoutDecisionMaker,
    IWorkoutPlanGenerator,
)
from fitai.domain.models import FitnessUserContext, InferenceResult, WorkoutPlan


class NeuralNetworkDecisionMaker:
    """Uses model output when available and delegates missing-output cases to rules."""

    def __init__(
        self,
        plan_generator: IWorkoutPlanGenerator,
        rule_based_plan_builder: IRuleBasedPlanBuilder,
    ) -> None:
        self._plan_generator = plan_generator
        self._rule_based_plan_builder = rule_based_plan_builder

    def create_plan(
        self,
        context: FitnessUserContext,
        result: InferenceResult | None,
    ) -> WorkoutPlan:
        if result is None:
            return self._rule_based_plan_builder.build(context)
        return self._plan_generator.generate(context, result)


class RuleBasedFallback:
    """Creates a rule-based plan and safely ignores optional model output."""

    def __init__(self, plan_builder: IRuleBasedPlanBuilder) -> None:
        self._plan_builder = plan_builder

    def create_plan(
        self,
        context: FitnessUserContext,
        result: InferenceResult | None,
    ) -> WorkoutPlan:
        return self._plan_builder.build(context)

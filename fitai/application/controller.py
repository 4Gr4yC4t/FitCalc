from dataclasses import replace
from datetime import datetime, timezone

from fitai.application.ports import (
    IDataQualityValidator,
    IFitCalcApiClient,
    IFeatureBuilder,
    IInferenceService,
    INutritionDataProvider,
    INutritionRecommendationService,
    IOvertrainingDetector,
    IProfileProvider,
    IRecommendationFormatter,
    ITrainingHistoryProvider,
    IWorkoutDecisionMaker,
    InferenceUnavailableError,
)
from fitai.domain.models import (
    FitnessRecommendation,
    FitnessUserContext,
    InferenceMetadata,
    ProModeSuggestion,
    RecommendationExplanation,
)


class FitAIController:
    """Coordinates recommendation services without constructing implementations."""

    def __init__(
        self,
        *,
        profile_provider: IProfileProvider,
        training_history_provider: ITrainingHistoryProvider,
        nutrition_data_provider: INutritionDataProvider,
        feature_builder: IFeatureBuilder,
        inference_service: IInferenceService,
        decision_maker: IWorkoutDecisionMaker,
        nutrition_service: INutritionRecommendationService,
        overtraining_detector: IOvertrainingDetector,
        formatter: IRecommendationFormatter,
        data_quality_validator: IDataQualityValidator,
        api_client: IFitCalcApiClient,
    ) -> None:
        self._profile_provider = profile_provider
        self._training_history_provider = training_history_provider
        self._nutrition_data_provider = nutrition_data_provider
        self._feature_builder = feature_builder
        self._inference_service = inference_service
        self._decision_maker = decision_maker
        self._nutrition_service = nutrition_service
        self._overtraining_detector = overtraining_detector
        self._formatter = formatter
        self._data_quality_validator = data_quality_validator
        self._api_client = api_client

    def build_recommendation(self, user_id: str) -> FitnessRecommendation:
        context = self._load_context(user_id)
        valid_data = self._data_quality_validator.validate(context)

        inference_result = None
        if valid_data:
            features = self._feature_builder.build(context)
            try:
                inference_result = self._inference_service.run(features)
            except InferenceUnavailableError:
                inference_result = None

        workout_plan = self._decision_maker.create_plan(context, inference_result)

        recommendation = FitnessRecommendation(
            workout_plan=workout_plan,
            nutrition=self._nutrition_service.recommend(context),
            overtraining_alert=self._overtraining_detector.detect(context),
            pro_mode_suggestion=self._pro_mode_suggestion(),
            explanation=RecommendationExplanation(text=""),
            metadata=(
                inference_result.metadata
                if inference_result is not None
                and inference_result.metadata is not None
                else InferenceMetadata(
                    model_version="unknown"
                    if inference_result is not None
                    else "fallback",
                    calculated_at=self._now(),
                )
            ),
        )
        recommendation = replace(
            recommendation,
            explanation=self._formatter.explain(recommendation),
        )
        self._api_client.post_recommendation(user_id, recommendation)
        return recommendation

    def _load_context(self, user_id: str) -> FitnessUserContext:
        return FitnessUserContext(
            user_id=user_id,
            profile=self._profile_provider.get_profile(user_id),
            goal=self._profile_provider.get_goal(user_id),
            training_history=self._training_history_provider.get_training_history(
                user_id
            ),
            nutrition_log=self._nutrition_data_provider.get_nutrition_log(user_id),
            body_weight_history=self._nutrition_data_provider.get_weight_history(
                user_id
            ),
        )

    @staticmethod
    def _pro_mode_suggestion() -> ProModeSuggestion:
        return ProModeSuggestion(
            suggested=False,
            reason="Критерии перехода в режим Профи ещё не заданы.",
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

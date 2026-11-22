from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from fitai.application.workout_plans import ExperienceLevel, WorkoutPlanService


class WorkoutPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ExperienceLevel = Field(
        description="Уровень подготовки.", examples=["beginner"]
    )


class WorkoutPlanResponse(BaseModel):
    level: ExperienceLevel = Field(examples=["beginner"])
    exercises: list[str] = Field(
        description="Упражнения в порядке выполнения.",
        examples=[["Ходьба — 10 минут", "Приседания к стулу — 2 × 8"]],
    )
    intensity: float = Field(
        ge=0, le=1, description="Условная интенсивность от 0 до 1.", examples=[0.35]
    )


class ExperienceLevelResponse(BaseModel):
    level: ExperienceLevel = Field(examples=["beginner"])
    label: str = Field(description="Название для интерфейса.", examples=["Начинающий"])


def create_api_router(
    plan_service: WorkoutPlanService, level_labels: dict[ExperienceLevel, str]
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["Тренировки"])

    @router.get(
        "/experience-levels",
        summary="Получить уровни подготовки",
        description="Возвращает уровни подготовки, доступные при запросе плана.",
    )
    def list_experience_levels() -> list[ExperienceLevelResponse]:
        return [
            ExperienceLevelResponse(level=level, label=label)
            for level, label in level_labels.items()
        ]

    @router.post(
        "/workout-plans",
        summary="Рассчитать план тренировки",
        responses={500: {"description": "Внутренняя ошибка при расчёте плана."}},
        description=(
            "Возвращает общий план по уровню подготовки. "
            "План рассчитывается при каждом запросе и не сохраняется."
        ),
    )
    def calculate_workout_plan(request: WorkoutPlanRequest) -> WorkoutPlanResponse:
        plan = plan_service.plan_for_level(request.level)
        return WorkoutPlanResponse(
            level=request.level,
            exercises=list(plan.exercises),
            intensity=plan.intensity,
        )

    return router

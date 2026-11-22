from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fitai.application.decision_makers import RuleBasedFallback
from fitai.application.workout_plans import (
    ExperienceLevel,
    ExperiencePlanBuilder,
    WorkoutPlanService,
)
from fitai.domain.models import WorkoutPlan
from fitai.web.api import create_api_router

LEVEL_LABELS = {
    ExperienceLevel.BEGINNER: "Начинающий",
    ExperienceLevel.INTERMEDIATE: "Средний",
    ExperienceLevel.ADVANCED: "Продвинутый",
}


def create_app(plan_service: WorkoutPlanService | None = None) -> FastAPI:
    app = FastAPI(
        title="FitCalc",
        version="1.0.0",
        description="Планы тренировок по уровню подготовки.",
    )
    if plan_service is None:
        plan_service = WorkoutPlanService(RuleBasedFallback(ExperiencePlanBuilder()))
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

    def render(
        request: Request,
        *,
        level: ExperienceLevel = ExperienceLevel.BEGINNER,
        plan: WorkoutPlan | None = None,
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "levels": LEVEL_LABELS,
                "selected_level": level,
                "plan": plan,
                "error": error,
            },
            status_code=status_code,
        )

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return render(request)

    @app.post("/", response_class=HTMLResponse)
    def show_plan(
        request: Request,
        level: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        try:
            experience = ExperienceLevel(level)
        except ValueError:
            return render(
                request, error="Выберите уровень подготовки из списка.", status_code=422
            )
        return render(
            request, level=experience, plan=plan_service.plan_for_level(experience)
        )

    app.include_router(create_api_router(plan_service, LEVEL_LABELS))
    return app


app = create_app()

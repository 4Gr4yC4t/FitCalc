import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from fitai.application.workout_plans import ExperienceLevel, WorkoutPlanService
from fitai.domain.models import WorkoutPlan
from fitai.web.app import app, create_app


class WorkoutPlanServiceTest(unittest.TestCase):
    def test_delegates_to_the_injected_strategy(self):
        strategy = Mock()
        strategy.create_plan.return_value = WorkoutPlan(("Ходьба",), 0.4)
        service = WorkoutPlanService(strategy)

        self.assertEqual(
            service.plan_for_level(ExperienceLevel.INTERMEDIATE),
            strategy.create_plan.return_value,
        )
        context, inference_result = strategy.create_plan.call_args.args
        self.assertEqual(context.profile.experience_level, "intermediate")
        self.assertIsNone(inference_result)


class WorkoutFormTest(unittest.TestCase):
    def test_home_lists_all_levels(self):
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        for level in ExperienceLevel:
            self.assertIn(f'value="{level.value}"', response.text)

    def test_each_level_renders_its_plan(self):
        with TestClient(app) as client:
            for level, exercise in (
                ("beginner", "Приседания к стулу"),
                ("intermediate", "Отжимания от опоры"),
                ("advanced", "Боковая планка"),
            ):
                with self.subTest(level=level):
                    response = client.post("/", data={"level": level})
                    self.assertEqual(response.status_code, 200)
                    self.assertIn(exercise, response.text)
                    self.assertIn(f'value="{level}" selected', response.text)

    def test_invalid_input_renders_an_error_without_calling_the_service(self):
        service = Mock(spec=WorkoutPlanService)
        with TestClient(create_app(service)) as client:
            for form_data in ({}, {"level": "unknown"}):
                response = client.post("/", data=form_data)
                self.assertEqual(response.status_code, 422)
                self.assertIn("Выберите уровень подготовки из списка.", response.text)
        service.plan_for_level.assert_not_called()

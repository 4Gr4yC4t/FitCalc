import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from fitai.application.workout_plans import ExperienceLevel, WorkoutPlanService
from fitai.domain.models import WorkoutPlan
from fitai.web.app import app, create_app


class WorkoutApiTest(unittest.TestCase):
    def test_levels_list_matches_supported_levels(self):
        with TestClient(app) as client:
            response = client.get("/api/v1/experience-levels")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["level"] for item in response.json()],
            [level.value for level in ExperienceLevel],
        )
        self.assertEqual(response.json()[0]["label"], "Начинающий")

    def test_plans_use_the_application_service(self):
        service = Mock(spec=WorkoutPlanService)
        service.plan_for_level.return_value = WorkoutPlan(("Ходьба",), 0.4)
        with TestClient(create_app(service)) as client:
            response = client.post("/api/v1/workout-plans", json={"level": "advanced"})

        service.plan_for_level.assert_called_once_with(ExperienceLevel.ADVANCED)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "level": "advanced", "exercises": ["Ходьба"], "intensity": 0.4,
        })

    def test_all_supported_levels_return_distinct_plans(self):
        with TestClient(app) as client:
            responses = [
                client.post("/api/v1/workout-plans", json={"level": level.value})
                for level in ExperienceLevel
            ]

        self.assertTrue(all(response.status_code == 200 for response in responses))
        plans = {tuple(response.json()["exercises"]) for response in responses}
        self.assertEqual(len(plans), len(ExperienceLevel))

    def test_invalid_requests_return_structured_validation_errors(self):
        service = Mock(spec=WorkoutPlanService)
        cases = (
            ({}, "missing"),
            ({"level": "unknown"}, "enum"),
            ({"level": 3}, "enum"),
            ({"level": "advanced", "weight": 75}, "extra_forbidden"),
        )
        with TestClient(create_app(service)) as client:
            for payload, error_type in cases:
                with self.subTest(payload=payload):
                    response = client.post("/api/v1/workout-plans", json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["detail"][0]["type"], error_type)
            malformed = client.post(
                "/api/v1/workout-plans",
                content="{",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(malformed.status_code, 422)
            self.assertEqual(malformed.json()["detail"][0]["type"], "json_invalid")

        service.plan_for_level.assert_not_called()

    def test_plan_service_failure_returns_internal_server_error(self):
        service = Mock(spec=WorkoutPlanService)
        service.plan_for_level.side_effect = RuntimeError("internal service details")
        with TestClient(create_app(service), raise_server_exceptions=False) as client:
            response = client.post("/api/v1/workout-plans", json={"level": "beginner"})

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal service details", response.text)

    def test_openapi_contains_the_contract_and_documentation_pages(self):
        with TestClient(app) as client:
            response = client.get("/openapi.json")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Swagger UI", client.get("/docs").text)
            self.assertIn("ReDoc", client.get("/redoc").text)

        schema = response.json()
        self.assertIn("get", schema["paths"]["/api/v1/experience-levels"])
        operation = schema["paths"]["/api/v1/workout-plans"]["post"]
        self.assertEqual(set(operation["responses"]), {"200", "422", "500"})
        self.assertEqual(
            operation["requestBody"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/WorkoutPlanRequest",
        )
        self.assertEqual(
            operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/WorkoutPlanResponse",
        )
        models = schema["components"]["schemas"]
        self.assertEqual(
            models["ExperienceLevel"]["enum"],
            [level.value for level in ExperienceLevel],
        )
        self.assertEqual(models["WorkoutPlanResponse"]["properties"]["intensity"]["maximum"], 1)

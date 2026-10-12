import unittest
from dataclasses import dataclass
from typing import Any

import numpy

from fitai.application.ports import InferenceUnavailableError
from fitai.domain.models import FeatureVector
from fitai.infrastructure.onnx_inference_service import OnnxInferenceService


@dataclass
class NodeMetadata:
    name: str


class FakeSession:
    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = outputs or [numpy.array([[0.2, 0.8]], dtype=numpy.float32)]
        self.input_feed: dict[str, Any] | None = None
        self.closed = False
        self.run_error: Exception | None = None

    def get_inputs(self) -> list[NodeMetadata]:
        return [NodeMetadata("features")]

    def get_outputs(self) -> list[NodeMetadata]:
        return [NodeMetadata("scores")]

    def run(self, output_names: None, input_feed: dict[str, Any]) -> list[Any]:
        if self.run_error is not None:
            raise self.run_error
        self.input_feed = input_feed
        return self.outputs

    def close(self) -> None:
        self.closed = True


class OnnxInferenceServiceTests(unittest.TestCase):
    def test_run_passes_one_float32_row_and_returns_flat_scores(self) -> None:
        session = FakeSession()
        service = OnnxInferenceService(session)

        result = service.run(FeatureVector((1, 2.5, -3)))

        numpy.testing.assert_allclose(result.scores, [0.2, 0.8], rtol=1e-6)
        self.assertIsNotNone(session.input_feed)
        tensor = session.input_feed["features"]
        self.assertEqual(tensor.shape, (1, 3))
        self.assertEqual(tensor.dtype, numpy.float32)
        numpy.testing.assert_array_equal(tensor, [[1, 2.5, -3]])

    def test_context_manager_closes_owned_session(self) -> None:
        session = FakeSession()

        with OnnxInferenceService(session):
            self.assertFalse(session.closed)

        self.assertTrue(session.closed)

    def test_close_is_idempotent_and_prevents_future_inference(self) -> None:
        session = FakeSession()
        service = OnnxInferenceService(session)

        service.close()
        service.close()

        self.assertTrue(session.closed)
        with self.assertRaisesRegex(RuntimeError, "session is closed"):
            service.run(FeatureVector((1,)))

    def test_empty_feature_vector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            FeatureVector(())

    def test_session_errors_are_not_hidden(self) -> None:
        session = FakeSession()
        service = OnnxInferenceService(session)
        failure = RuntimeError("model execution failed")
        session.run_error = failure

        with self.assertRaisesRegex(InferenceUnavailableError, "ONNX model"):
            service.run(FeatureVector((1,)))

    def test_multiple_model_outputs_are_rejected(self) -> None:
        session = FakeSession(outputs=[numpy.array([1]), numpy.array([2])])
        service = OnnxInferenceService(session)

        with self.assertRaisesRegex(RuntimeError, "Expected one ONNX output"):
            service.run(FeatureVector((1,)))


if __name__ == "__main__":
    unittest.main()

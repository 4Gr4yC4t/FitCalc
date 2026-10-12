from os import PathLike
from typing import Any, Protocol, Sequence

from fitai.domain.models import FeatureVector, InferenceResult


class _NodeMetadata(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_NodeMetadata]: ...

    def get_outputs(self) -> Sequence[_NodeMetadata]: ...

    def run(self, output_names: None, input_feed: dict[str, Any]) -> Sequence[Any]: ...

    def close(self) -> None: ...


class OnnxInferenceService:
    """Runs a single-input, single-output ONNX model and owns its session."""

    def __init__(self, session: _InferenceSession) -> None:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(
                f"Expected an ONNX model with exactly one input, got {len(inputs)}"
            )
        if len(outputs) != 1:
            raise ValueError(
                f"Expected an ONNX model with exactly one output, got {len(outputs)}"
            )
        self._session = session
        self._input_name = inputs[0].name
        self._closed = False

    @classmethod
    def from_model(
        cls,
        model_path: str | PathLike[str],
    ) -> "OnnxInferenceService":
        """Create a CPU-backed ONNX session for the given model file."""
        try:
            import onnxruntime
        except ImportError as error:
            raise RuntimeError(
                "ONNX inference requires onnxruntime; install it with "
                "`python3 -m pip install onnxruntime`"
            ) from error

        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        return cls(session)

    def run(self, features: FeatureVector) -> InferenceResult:
        """Run the model once with one row of float32 features."""
        if self._closed:
            raise RuntimeError("Cannot run inference after the ONNX session is closed")

        import numpy

        tensor = numpy.asarray([features.values], dtype=numpy.float32)
        outputs = self._session.run(None, {self._input_name: tensor})
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected one ONNX output tensor, got {len(outputs)}"
            )
        scores = numpy.asarray(outputs[0]).reshape(-1).tolist()
        return InferenceResult(tuple(scores))

    def close(self) -> None:
        """Close the owned ONNX session; repeated calls are safe."""
        if not self._closed:
            self._session.close()
            self._closed = True

    def __enter__(self) -> "OnnxInferenceService":
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed ONNX inference service")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

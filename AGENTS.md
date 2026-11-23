# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project overview

FitCalc is a workout and nutrition planning application. The repository contains two
related parts:

- **FitCalc** — the user-facing FastAPI web application (HTML form + REST API) that
  returns a workout plan for a selected experience level.
- **FitAI** — an "intelligent" module (`fitai/`) that is intended to personalize
  workouts and nutrition from a user profile, training history, nutrition log and
  weight history, optionally using an ONNX model with a rule-based fallback.

This is a **student semester/coursework project** (Software Engineering course), not a
production product. See `disclaimer.md`. Most user-facing documentation is written in
Russian (`README.md`, `tz.md`, `docs/*.md`, `fitai_docs/*.md`), while code, docstrings
and identifiers are in English.

## Tech stack

- Python 3.11, FastAPI, Uvicorn, Jinja2, Pydantic v2
- SQLAlchemy 2.x + Alembic (SQLite persistence for the FitAI module)
- `onnxruntime` + `numpy` (optional; the model file is not committed)
- `matplotlib` (lab demo plotting), `httpx` (tests), `unittest` (no pytest)

## Project layout

```
fitai/
  domain/           Frozen dataclasses (models.py) and storage contracts (repository.py)
  application/      FitAIController, service ports (Protocols), strategy decision makers,
                    workout-plan builders
  infrastructure/   ONNX inference adapter, SQLAlchemy ORM/repository, data mappers,
                    DB init/migrations, buffered metrics writer
  web/              FastAPI app factory (app.py), REST API (api.py), Jinja2 template
  lab4_demo.py      Persistence/metrics demo (simulated, not real inference)
migrations/         Alembic migrations (SQLite schema for the FitAI module)
tests/              unittest suite (discovered from the repo root)
docs/               Lab reports (LAB5.md, LAB7.md)
fitai_docs/         FitAI design docs, PlantUML diagrams, lab4 report
```

## Common commands

All commands run from the repository root.

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Run the web app (no DB, account or ONNX model needed for the plan form/API)
python -m uvicorn fitai.web.app:app

# Run the full test suite
python -m unittest discover -s tests -v

# Run the lab4 persistence demo (100 iterations, writes an FPS/memory plot)
python -m fitai.lab4_demo --iterations 100 --plot fps_vs_gc.png
# ...with an isolated database:
python -m fitai.lab4_demo --database /tmp/fitai-lab4.sqlite3
```

`onnxruntime` and `numpy` are only required to run a real ONNX model and are not in
`requirements.txt`. The ONNX model file is intentionally not committed; the web app and
tests run without it.

## Architecture and conventions

The `fitai` module follows a layered (clean) architecture with dependency injection:

- **Domain layer** (`fitai/domain/`) holds plain frozen dataclasses
  (`FitnessUserContext`, `WorkoutPlan`, `InferenceResult`, `AiAgent`, etc.) and
  `Protocol`-based storage contracts. It must not depend on FastAPI, SQLAlchemy or
  ONNX.
- **Application layer** (`fitai/application/`) defines service **ports** as
  `typing.Protocol`s in `ports.py`. `FitAIController` coordinates providers, feature
  building, inference, decision making, nutrition and formatting; it receives all
  collaborators as keyword-only constructor arguments and never constructs them.
- **Strategy pattern**: `IWorkoutDecisionMaker` is implemented by
  `NeuralNetworkDecisionMaker` (uses model output, delegates `None` results to rules)
  and `RuleBasedFallback` (rules only). Both satisfy the same `create_plan(context,
  result)` contract, so the controller and `WorkoutPlanService` can call either
  identically.
- **Infrastructure layer** (`fitai/infrastructure/`) provides adapters:
  `OnnxInferenceService` (single-input/single-output, owns its session, context-manager
  closes it), `SqlAgentRepository` + `data_mapper.py` (ORM rows <-> domain models),
  `db.py` (SQLite engine + Alembic migration), and `BufferedMetricsWriter` (background
  thread that batches metrics to the repository).
- **Web layer** (`fitai/web/`) exposes a form (`GET /`, `POST /`) and a JSON API
  (`GET /api/v1/experience-levels`, `POST /api/v1/workout-plans`). `create_app()` is a
  factory that assembles the app with default implementations and accepts a
  `WorkoutPlanService` for test injection.

Key design principles reflected in the code and tests:

- **Dependency inversion** — controllers/services depend on `Protocol`s, not concrete
  classes.
- **ISP** — data providers are split into `IProfileProvider`,
  `ITrainingHistoryProvider`, `INutritionDataProvider` instead of one fat interface
  (see `tests/test_isp.py`).
- **LSP** — decision-makers are interchangeable behind one contract.
- **Repositories** — one short session per operation; `SqlAgentRepository` manages
  transactions and delegates row mapping to `data_mapper.py`.
- API request schemas use `extra="forbid"` (reject unknown fields).

## Testing

- Use the built-in `unittest` framework (no pytest). Tests live in `tests/` and are
  discovered from the repository root.
- Tests cover the controller, decision makers, ONNX adapter, ISP contracts, web form,
  REST API, and SQLite persistence. When adding behavior, follow the existing
  `unittest.TestCase` + `unittest.mock.Mock` style.
- The app is imported as `from fitai.web.app import app` (module-level instance) and
  `create_app` (factory for injected services).

## Data and persistence notes

- The FitAI module uses SQLite. The default database path is
  `$XDG_DATA_HOME/fitcalc/fitai.sqlite3`, falling back to
  `~/.local/share/fitcalc/fitai.sqlite3`. Use `--database` (lab demo) or
  `initialize_database(path)` to override.
- Schema is managed by Alembic (`migrations/versions/0001_initial_schema.py`); apply
  with `migrate_database` or `initialize_database`.
- `player_progress.inventory_json` deliberately stores an inventory as a JSON column
  (a documented 1NF exception); if item-level queries are needed, extract it to an
  `inventory_items` table.
- `BufferedMetricsWriter` must be closed (or used as a context manager) to flush the
  remaining queue; hard-killing the process can lose buffered metrics.

## Language and style

- Code, docstrings, comments and identifiers are in **English**.
- User-facing text, HTML templates and lab/design documents are in **Russian**.
- Keep new code consistent with the existing layered structure and `unittest` style.

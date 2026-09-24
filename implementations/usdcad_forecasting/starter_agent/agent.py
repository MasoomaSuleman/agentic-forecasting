"""USD/CAD starter agent for the agentic forecasting experiment.

The target is a forward cumulative log return of FRED DEXCAUS, with optional
FRED macro covariates and cutoff-aware web research.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput, build_adk_agent
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, CodeExecutionConfig, ContextRetrievalConfig
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel

_SKILLS_ROOT = Path(__file__).parent / "skills"
_FORECASTING_SKILL = _SKILLS_ROOT / "forecasting"
_RESEARCH_SKILL = _SKILLS_ROOT / "research-playbook"
_CODE_ANALYSIS_SKILL = _SKILLS_ROOT / "code-analysis-playbook"


class UsdcadStarterPromptBuilder(BaseModel):
    """Serialize USD/CAD history and a leak-safe covariate snapshot."""
    model_config = {"extra": "forbid"}
    history: int = 64
    covariate_series_ids: list[str] = []

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id).tail(self.history)
        rows = ["date,log_return"]
        rows.extend(
            f"{pd.Timestamp(ts).date()},{float(value):.8f}"
            for ts, value in zip(df["timestamp"], df["value"])
        )
        covariate_snapshot: dict[str, float] = {}
        for series_id in self.covariate_series_ids:
            try:
                cov_df = context.get_series(series_id)
            except Exception:
                continue
            if not cov_df.empty:
                covariate_snapshot[series_id] = round(float(cov_df["value"].iloc[-1]), 8)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_log_return": float(df["value"].iloc[-1]),
                "last_date": str(pd.Timestamp(df["timestamp"].iloc[-1]).date()),
                "n_obs": int(len(df)),
            },
            "target_history_csv": "\n".join(rows),
        }
        if covariate_snapshot:
            payload["covariate_snapshot"] = covariate_snapshot
        return json.dumps(payload, indent=2)


def _build_starter_instruction() -> str:
    return (
        "## Role\n\n"
        "You are a foreign-exchange analyst specializing in USD/CAD. You understand "
        "Bank of Canada and Federal Reserve policy paths, US-Canada rate differentials, "
        "Canadian and US macro data, WTI/oil, commodity sensitivity, risk sentiment, "
        "and CAD-specific drivers. Keep reasoning transparent and avoid false precision.\n\n"
        "## How to respond\n\n"
        "- For open-ended questions, answer directly and concisely; do not ask for a JSON payload.\n"
        "- For a structured probabilistic forecast, use the forecasting skill and produce a calibrated distribution.\n"
        "- DEXCAUS is Canadian dollars per US dollar: positive return means USD/CAD rises and USD strengthens relative to CAD.\n"
        "- Separate observed evidence from interpretation and never use information after the forecast cutoff.\n"
        "- When submitting a structured forecast, call set_model_response exactly once with a json_response containing valid JSON that matches output_schema.\n"
        "- The json_response value must be raw JSON, not a JSON-encoded string inside another JSON object. Do not add markdown fences, comments, trailing commas, or backslash-escaped punctuation.\n"
        "- Use double quotes for every JSON key and string. Keep quantiles as the exact structure specified by output_schema."
    )

_STARTER_INSTRUCTION = _build_starter_instruction()

_CONTEXT_RETRIEVAL_INSTRUCTION = """You are a USD/CAD market-intelligence specialist with web search.

When research is useful, summarize the current state of:
- Bank of Canada policy, guidance, and Canadian rate expectations;
- Federal Reserve policy, guidance, and US rate expectations;
- US-Canada yield/rate differentials;
- Canadian and US inflation, employment, and growth;
- WTI/oil and other commodity developments relevant to CAD;
- broad risk sentiment and material geopolitical or trade-policy shocks.

Use focused queries and prefer primary releases and high-quality financial reporting.
Ground factual claims in retrieved results. When a cutoff date is supplied, never use
events after it. For historical origins, discard results that cannot be verified as
pre-cutoff rather than filling gaps from background knowledge.
"""


def build_starter_agent_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    enable_search: bool = True,
    enable_code_exec: bool = False,
) -> AgentConfig:
    """Build the USD/CAD starter AgentConfig."""
    skills_dirs: list[Path] = [_FORECASTING_SKILL]
    if enable_search:
        skills_dirs.append(_RESEARCH_SKILL)
    if enable_code_exec:
        skills_dirs.append(_CODE_ANALYSIS_SKILL)

    context_retrieval = (
        ContextRetrievalConfig(
            enabled=True,
            instruction=_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        )
        if enable_search
        else ContextRetrievalConfig()
    )
    return AgentConfig(
        name="usdcad_starter_agent",
        model=model,
        instruction=_STARTER_INSTRUCTION,
        max_output_tokens=16_384 if enable_code_exec else None,
        context_retrieval=context_retrieval,
        code_execution=CodeExecutionConfig(enabled=enable_code_exec),
        skills_dirs=skills_dirs,
    )


class _StarterForecastPromptBuilder:
    """Add the forecast directive and exact output schema to the task payload."""
    def __init__(self, inner: Callable[..., str], output_schema_json: str) -> None:
        self._inner = inner
        self._schema_json = output_schema_json

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(self._inner(task=task, context=context))
        payload["instructions"] = (
            "Produce a calibrated probabilistic forecast for this task and return it "
            "by calling set_model_response with a json_response string matching output_schema exactly. "
            "The json_response itself must be valid JSON: use double quotes, no markdown fences, "
            "no comments, no trailing commas, and no backslash-escaped punctuation. Do not wrap "
            "the JSON in another JSON object or return it as a Python/JSON repr."
        )
        payload["output_schema"] = self._schema_json
        return json.dumps(payload, indent=2)


class _RetryingAgentPredictor:
    """Retry transient malformed structured-output responses."""

    def __init__(self, predictor: AgentPredictor, max_retries: int = 2, retry_delay: float = 1.0) -> None:
        self._predictor = predictor
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def predict(self, task: ForecastingTask, context: ForecastContext):
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._predictor.predict(task, context)
            except json.JSONDecodeError:
                if attempt == attempts:
                    raise
                print(
                    f"    Structured-output JSON parse failed; retrying "
                    f"({attempt}/{self._max_retries})..."
                )
                time.sleep(self._retry_delay * attempt)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._predictor, name)


def build_starter_agent_predictor(
    config: AgentConfig,
    *,
    covariate_series_ids: list[str] | None = None,
) -> AgentPredictor:
    """Create a resilient AgentPredictor for USD/CAD probabilistic forecasting."""
    predictor = AgentPredictor(
        agent_config=config,
        prompt_builder=_StarterForecastPromptBuilder(
            UsdcadStarterPromptBuilder(covariate_series_ids=covariate_series_ids or []),
            ContinuousAgentForecastOutput.prompt_schema_json(),
        ),
        output_schema=ContinuousAgentForecastOutput,
    )
    return _RetryingAgentPredictor(predictor)


def __getattr__(name: str) -> Any:
    """Expose root_agent lazily for ADK interactive use."""
    if name == "root_agent":
        return build_adk_agent(build_starter_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Predictor that uses an ADK agent for forecasting."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar, cast

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.langfuse_traces import stamp_forecast_on_trace
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY, AgentConfig, build_adk_agent
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence, trace_url_for
from google.adk.agents.base_agent import BaseAgent
from pydantic import ValidationError

logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


def _run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from the sync ``Predictor`` interface."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: T | None = None
    error: BaseException | None = None

    def run_in_thread() -> None:
        nonlocal error, result
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
        except BaseException as exc:
            error = exc
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return cast("T", result)


class ForecastPromptBuilder(Protocol):
    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        ...


class AgentPredictor(Predictor):
    """Predictor that drives an ADK agent and validates structured forecasts."""

    def __init__(
        self,
        agent_config: AgentConfig,
        prompt_builder: ForecastPromptBuilder,
        *,
        output_schema: type[AgentForecastOutput],
        enable_langfuse_tracing: bool | None = None,
        runner: AdkTextRunner | None = None,
        max_output_retries: int = 2,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if enable_langfuse_tracing is None:
            try:
                import langfuse  # noqa: F401, PLC0415
                enable_langfuse_tracing = True
            except ModuleNotFoundError:
                enable_langfuse_tracing = False

        self.prompt_builder = prompt_builder
        self.agent_config = agent_config
        self.output_schema: type[AgentForecastOutput] = output_schema
        self.enable_langfuse_tracing = enable_langfuse_tracing
        self.max_output_retries = max(0, int(max_output_retries))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._forecast_output_modality = output_schema.modality

        if runner is None:
            built_agent = build_adk_agent(agent_config, output_schema=output_schema)
            self._agent: BaseAgent = built_agent
            self._runner = AdkTextRunner(
                agent=built_agent,
                config=AdkTextRunnerConfig(
                    app_name="agentic_forecasting_predictor",
                    default_user_id="forecasting_agent",
                    fresh_session_per_message=True,
                    enable_langfuse_tracing=self.enable_langfuse_tracing,
                    langfuse_tags=["agent_predictor", "track1"],
                    langfuse_trace_name=self.predictor_id,
                    langfuse_propagate_metadata={
                        "predictor_id": self.predictor_id,
                        "agent_name": built_agent.name,
                        "model": str(built_agent.model),
                        "output_modality": self._forecast_output_modality,
                    },
                ),
            )
        else:
            self._runner = runner
            self._agent = runner.agent

    @property
    def predictor_id(self) -> str:
        model = getattr(self._agent, "model", None)
        if not isinstance(model, str):
            inner = getattr(model, "model", None)
            model = inner if isinstance(inner, str) else None
        model_suffix = f"_{model.rsplit('/', 1)[-1]}" if model else ""
        return f"agent_predictor_{self._agent.name}{model_suffix}_{self._forecast_output_modality}"

    def _run_and_parse(self, prompt: str, initial_state: dict[str, str]) -> AgentForecastOutput:
        """Run the agent once and parse its structured response."""
        output_str = _run_coroutine_sync(
            self._runner.run_text_async(prompt, initial_state=initial_state)
        )
        output_str = strip_markdown_fence(output_str or "")
        try:
            return self.output_schema.model_validate_json(output_str)
        except ValidationError:
            try:
                return self.output_schema.model_validate(json.loads(output_str))
            except Exception:
                logger.warning(
                    "Raw agent response (schema validation failed):\n%s",
                    output_str,
                )
                raise

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        prompt = self.prompt_builder(task=task, context=context)
        initial_state = {AS_OF_STATE_KEY: str(context.as_of)[:10]}

        output: AgentForecastOutput | None = None
        last_error: Exception | None = None

        # LLM structured-output failures can be transient (empty response,
        # malformed JSON, or a one-off schema violation). Retry the whole
        # model call with a fresh ADK session rather than repairing the text.
        for attempt in range(self.max_output_retries + 1):
            try:
                output = self._run_and_parse(prompt, initial_state)
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt >= self.max_output_retries:
                    raise
                logger.warning(
                    "Structured forecast parse failed; retrying (%d/%d).",
                    attempt + 1,
                    self.max_output_retries,
                )
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))

        if output is None:
            raise RuntimeError("Agent did not produce a valid structured forecast") from last_error

        try:
            predictions = output.to_predictions(
                task=task,
                context=context,
                predictor_id=self.predictor_id,
            )
        except Exception as e:
            logger.error("Error converting output to list of predictions: %s", e)
            return []

        trace_id = self._runner.last_trace_id
        if trace_id is not None:
            trace_url = trace_url_for(trace_id)
            for prediction in predictions:
                prediction.metadata.setdefault("langfuse_trace_id", trace_id)
                if trace_url is not None:
                    prediction.metadata.setdefault("langfuse_trace_url", trace_url)
            stamp_forecast_on_trace(predictions, trace_id=trace_id)

        return predictions

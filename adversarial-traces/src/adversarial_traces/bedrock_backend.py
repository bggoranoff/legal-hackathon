"""Optional Amazon Bedrock transport (Converse API); no SDK import until needed.

Structured output is obtained by forcing a single tool whose input schema is the
requested JSON schema ("tool" mode, the default). For models that do not
support forced tool choice, "text" mode asks for bare JSON and parses it.

Bedrock has no built-in web search, so requests with ``web_search=True`` are
refused. Use ``OpenAIResponsesBackend`` for a web-searching final attacker.

Authentication follows boto3's normal chain, including
``AWS_BEARER_TOKEN_BEDROCK``. The region comes from ``region`` or
``AWS_REGION`` / ``AWS_DEFAULT_REGION``.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from .models import JSONRequest, JSONResponse, ModelResponseError, TraceError

_MODES = ("tool", "text")
# Reasoning levels accepted by OpenAI reasoning models on Bedrock (e.g. GPT 6 Luna).
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
# GPT 6 Luna runs at max reasoning unless told otherwise. Other models get no
# reasoning setting by default (Claude on Bedrock rejects it).
DEFAULT_REASONING = {"gpt-6-luna": "max"}
_AUTO = "auto"


def default_reasoning_effort(model: str) -> str | None:
    """The reasoning level used when none is given: max for GPT 6 Luna, else none."""
    for name, effort in DEFAULT_REASONING.items():
        if name in model:
            return effort
    return None
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON number")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _parse_json_text(text: str) -> Any:
    """Parse a JSON object, tolerating a Markdown code fence around it."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    return json.loads(text, parse_constant=_reject_constant,
                      parse_float=_finite_float, object_pairs_hook=_unique_object)


def _error_code(exc: Exception) -> str:
    """A provider error code or class name; never the message, which may echo input."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,64}", code):
            return code
    return type(exc).__name__


class BedrockConverseBackend:
    """Fresh, stateless Converse calls with a caller-selected Bedrock model ID."""

    def __init__(self, model: str, *, client: Any = None, region: str | None = None,
                 max_output_tokens: int | None = None, timeout: float | None = None,
                 structured: str = "tool", reasoning_effort: str | None = _AUTO):
        if type(model) is not str or not model.strip():
            raise TraceError("An explicit model name is required")
        if reasoning_effort == _AUTO:
            reasoning_effort = default_reasoning_effort(model) if isinstance(model, str) else None
        # Reasoning tokens count toward the output limit and take longer, so
        # reasoning runs get more room unless the caller sets limits.
        thinking = reasoning_effort not in (None, "none")
        if max_output_tokens is None:
            max_output_tokens = 32000 if thinking else 8192
        if timeout is None:
            timeout = 600.0 if thinking else 120.0
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise TraceError("max_output_tokens must be a positive integer")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise TraceError("timeout must be a positive finite number")
        if structured not in _MODES:
            raise TraceError("structured must be 'tool' or 'text'")
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise TraceError("reasoning_effort must be one of: " + ", ".join(REASONING_EFFORTS))
        self.model_id = model
        self._client = client
        self.region = region
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.structured = structured
        # By default: max for GPT 6 Luna, unset for other models. None leaves the
        # model's own default. Only OpenAI-style reasoning models accept a level.
        self.reasoning_effort = reasoning_effort

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ModelResponseError(
                    "Install the optional provider dependency: pip install 'adversarial-traces[bedrock]'"
                ) from None
            region = (self.region or os.environ.get("AWS_REGION")
                      or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
            try:
                self._client = boto3.client(
                    "bedrock-runtime", region_name=region,
                    config=Config(read_timeout=self.timeout, retries={"max_attempts": 3, "mode": "standard"}),
                )
            except Exception:
                raise ModelResponseError("Could not initialize the Bedrock client") from None
        return self._client

    def complete(self, request: JSONRequest) -> JSONResponse:
        if request.web_search:
            raise ModelResponseError(
                "Bedrock backend has no web search; use OpenAIResponsesBackend for web-search attacks"
            )
        if not _TOOL_NAME.fullmatch(request.schema_name):
            raise ModelResponseError("Schema name is not a valid Bedrock tool name")
        try:
            encoded_payload = json.dumps(request.payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            raise ModelResponseError("Request payload is not finite JSON") from None

        system = request.system
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": encoded_payload}]}],
            "inferenceConfig": {"maxTokens": self.max_output_tokens},
        }
        if self.reasoning_effort is not None:
            kwargs["additionalModelRequestFields"] = {"reasoning": {"effort": self.reasoning_effort}}
        if self.structured == "tool":
            kwargs["toolConfig"] = {
                "tools": [{"toolSpec": {
                    "name": request.schema_name,
                    "description": "Submit the requested JSON answer.",
                    "inputSchema": {"json": request.schema},
                }}],
                "toolChoice": {"tool": {"name": request.schema_name}},
            }
            system += f" Submit the answer by calling the {request.schema_name} tool."
        else:
            system += (" Reply with ONLY one JSON object matching this JSON schema, with no "
                       "prose and no code fence: " + json.dumps(request.schema))
        kwargs["system"] = [{"text": system}]

        try:
            response = self._get_client().converse(**kwargs)
        except ModelResponseError:
            raise
        except Exception as exc:
            raise ModelResponseError(f"Bedrock request failed ({_error_code(exc)})") from None

        stop = response.get("stopReason") if isinstance(response, dict) else None
        if stop not in ("end_turn", "tool_use", "stop_sequence"):
            raise ModelResponseError("Bedrock returned an incomplete or filtered response")
        content = (response.get("output") or {}).get("message", {}).get("content")
        if not isinstance(content, list) or not content:
            raise ModelResponseError("Bedrock returned no output content")

        tool_inputs: list[Any] = []
        texts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise ModelResponseError("Bedrock returned malformed content")
            if "toolUse" in block:
                use = block["toolUse"]
                if not isinstance(use, dict) or use.get("name") != request.schema_name:
                    raise ModelResponseError("Bedrock called an unexpected tool")
                tool_inputs.append(use.get("input"))
            elif "text" in block:
                if type(block["text"]) is not str:
                    raise ModelResponseError("Bedrock returned malformed text")
                texts.append(block["text"])
            elif "reasoningContent" not in block:
                raise ModelResponseError("Bedrock returned an unexpected content block")

        if self.structured == "tool":
            if len(tool_inputs) != 1:
                raise ModelResponseError("Bedrock did not return exactly one structured answer")
            try:
                # Round-trip so the same finiteness/duplicate checks apply.
                data = _parse_json_text(json.dumps(tool_inputs[0], allow_nan=True))
            except (ValueError, TypeError):
                raise ModelResponseError("Bedrock returned invalid JSON") from None
        else:
            if tool_inputs or not texts:
                raise ModelResponseError("Bedrock returned no structured answer")
            try:
                data = _parse_json_text("".join(texts))
            except (ValueError, TypeError):
                raise ModelResponseError("Bedrock returned invalid JSON") from None
        if type(data) is not dict:
            raise ModelResponseError("Bedrock response JSON must be an object")
        # Schema-specific validation is repeated by each role adapter.
        return JSONResponse(data, web_search_used=False)

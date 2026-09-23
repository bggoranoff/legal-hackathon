"""Optional synchronous OpenAI Responses transport; no SDK import until needed.

Verified against official Responses structured-output and web-search guides:
https://developers.openai.com/api/docs/guides/structured-outputs
https://developers.openai.com/api/docs/guides/tools-web-search

``store=False`` avoids stored Responses state; it is not a promise of zero data
retention by a provider. Select models supporting the requested API capabilities.
"""
from __future__ import annotations

import json
import math
from typing import Any

from .models import JSONRequest, JSONResponse, ModelResponseError, TraceError


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


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


class OpenAIResponsesBackend:
    """Fresh JSON Responses calls with caller-selected model and optional client.

    No history, conversation ID, prior response ID, arbitrary request overrides,
    or text logging is kept by this adapter. A supplied client may implement its
    own logging/retry policy. SDK-created clients disable automatic retries.
    """

    def __init__(self, model: str, *, client: Any = None,
                 max_output_tokens: int = 8192, timeout: float = 60.0):
        if type(model) is not str or not model.strip():
            raise TraceError("An explicit model name is required")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise TraceError("max_output_tokens must be a positive integer")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise TraceError("timeout must be a positive finite number")
        self.model_id = model
        self._client = client
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ModelResponseError(
                    "Install the optional provider dependency: pip install 'adversarial-traces[openai]'"
                ) from None
            try:
                self._client = OpenAI(timeout=self.timeout, max_retries=0)
            except Exception:
                raise ModelResponseError("Could not initialize the OpenAI client") from None
        return self._client

    def complete(self, request: JSONRequest) -> JSONResponse:
        try:
            encoded_payload = json.dumps(request.payload, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            raise ModelResponseError("Request payload is not finite JSON") from None
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "store": False,
            "input": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": encoded_payload},
            ],
            "text": {"format": {
                "type": "json_schema", "name": request.schema_name,
                "strict": True, "schema": request.schema,
            }},
            "max_output_tokens": self.max_output_tokens,
        }
        if request.web_search:
            kwargs["tools"] = [{"type": "web_search"}]
            kwargs["tool_choice"] = "required"
        try:
            response = self._get_client().responses.create(**kwargs)
        except ModelResponseError:
            raise
        except Exception:
            raise ModelResponseError("OpenAI request failed") from None
        if (_field(response, "status") != "completed" or
                _field(response, "error") is not None or
                _field(response, "incomplete_details") is not None):
            raise ModelResponseError("OpenAI returned an incomplete or failed response")
        output = _field(response, "output")
        if not isinstance(output, (list, tuple)) or not output:
            raise ModelResponseError("OpenAI returned no output items")
        texts: list[str] = []
        completed_search = False
        for item in output:
            kind = _field(item, "type")
            if kind == "web_search_call":
                if _field(item, "status") != "completed":
                    raise ModelResponseError("OpenAI web search did not complete")
                action = _field(item, "action")
                if _field(action, "type") == "search":
                    completed_search = True
            elif kind == "message":
                if _field(item, "status") != "completed" or _field(item, "role") != "assistant":
                    raise ModelResponseError("OpenAI returned an incomplete output message")
                content = _field(item, "content")
                if not isinstance(content, (list, tuple)):
                    raise ModelResponseError("OpenAI returned malformed message content")
                for part in content:
                    if _field(part, "type") == "refusal":
                        raise ModelResponseError("OpenAI declined the model request")
                    if _field(part, "type") != "output_text" or type(_field(part, "text")) is not str:
                        raise ModelResponseError("OpenAI returned unexpected message content")
                    texts.append(_field(part, "text"))
            elif kind != "reasoning":
                # The only tool requested is built-in search; don't silently ignore
                # outstanding function calls or other unhandled actions.
                raise ModelResponseError("OpenAI returned an unexpected output item")
        if request.web_search and not completed_search:
            raise ModelResponseError("OpenAI response has no completed web search action")
        if not texts:
            raise ModelResponseError("OpenAI returned no structured answer")
        try:
            data = json.loads("".join(texts), parse_constant=_reject_constant,
                              parse_float=_finite_float, object_pairs_hook=_unique_object)
        except (ValueError, TypeError):
            raise ModelResponseError("OpenAI returned invalid JSON") from None
        if type(data) is not dict:
            raise ModelResponseError("OpenAI response JSON must be an object")
        # Schema-specific validation is deliberately repeated by each role adapter.
        return JSONResponse(data, web_search_used=completed_search)

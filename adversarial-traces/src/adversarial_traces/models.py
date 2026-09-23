"""Provider-independent data contracts. GroundTruth is evaluation-only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

JSON = Any
DEFAULT_ATTRIBUTES = ("matter_identity", "parties", "counsel", "amounts", "dates", "jurisdiction")


class TraceError(ValueError):
    """Invalid input trace, template, world, or configuration."""


class ModelResponseError(RuntimeError):
    """Malformed, incomplete, or unverifiable model response."""


@dataclass(frozen=True)
class Inference:
    attribute: str
    value: str | int | float | bool | None
    reasoning: str = ""
    certainty: float = 0.0
    spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hint:
    """Untrusted attacker feedback: rewrite guidance, never output text."""
    reasoning: str
    spans: tuple[str, ...] = ()
    segment_id: str | None = None


@dataclass(frozen=True)
class Segment:
    segment_id: str
    role: Literal["system", "user", "assistant", "tool"]
    kind: Literal["message", "tool_call", "tool_result"]
    payload: dict[str, JSON]
    tool_name: str | None = None
    call_id: str | None = None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class AbstractProfile:
    """Contains abstracted segments only; never originals, hints or ground truth."""
    trace: Trace
    placeholders: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticWorld:
    """Keys omit braces, e.g. BUYER -> 'Fictional Cedar Holdings'."""
    bindings: Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True)
class WorldRules:
    """Trusted caller-supplied coherence rules; values name binding keys."""
    date_orders: tuple[tuple[str, str], ...] = ()
    numeric_ranges: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    ratio_ranges: tuple[tuple[str, str, float, float], ...] = ()
    compatible_values: tuple[tuple[str, str, Mapping[str, Sequence[str]]], ...] = ()


@dataclass(frozen=True)
class MatterGuess:
    identity: str | None = None
    parties: tuple[str, ...] = ()
    certainty: float = 0.0


@dataclass(frozen=True)
class AttackReport:
    guesses: tuple[MatterGuess, ...]
    reasoning: str
    spans: tuple[str, ...] = ()
    web_search_used: bool = False
    completed: bool = True


@dataclass(frozen=True, repr=False)
class GroundTruth:
    """Explicit identity aliases and/or one alias group per required party."""
    identities: tuple[str, ...] = ()
    party_aliases: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class AttemptSummary:
    round_number: int
    status: str
    detail: str = ""


@dataclass(frozen=True)
class SynthesisResult:
    status: Literal["passed_attack", "failed"]
    trace: Trace | None
    attempts: tuple[AttemptSummary, ...]
    reason: str | None = None
    # Full attack reasoning, originals, identities and per-segment rewrites are
    # deliberately absent from the public result.

    @property
    def succeeded(self) -> bool:
        return self.status == "passed_attack"


class InferenceModel(Protocol):
    model_id: str

    def infer(self, text: str, attributes: tuple[str, ...]) -> Sequence[Inference]: ...


class AnonymizerModel(Protocol):
    def anonymize(self, text: str, inferences: tuple[Inference, ...], *,
                  hints: tuple[Hint, ...] = ()) -> str: ...


class WorldGenerator(Protocol):
    def generate(self, profile: AbstractProfile, rules: WorldRules) -> SyntheticWorld: ...


class FinalAttacker(Protocol):
    model_id: str

    def attack(self, trace: Trace, instruction: str) -> AttackReport: ...


class MatchJudge(Protocol):
    """Decides, without an answer key, whether an attack named the real matter.

    The judge is the only model that sees the original trace. Its verdict is a
    bool; nothing it writes is passed to any other model or returned.
    """
    model_id: str

    def judge(self, original: Trace, report: AttackReport) -> bool: ...


@dataclass(frozen=True)
class JSONRequest:
    """A fresh model request. No history or provider conversation ID."""
    system: str
    payload: dict[str, JSON]
    schema: dict[str, JSON]
    schema_name: str
    web_search: bool = False


@dataclass(frozen=True)
class JSONResponse:
    data: dict[str, JSON]
    web_search_used: bool = False


class JSONBackend(Protocol):
    model_id: str

    def complete(self, request: JSONRequest) -> JSONResponse: ...

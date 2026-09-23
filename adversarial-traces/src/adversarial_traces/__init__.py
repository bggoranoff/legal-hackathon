"""Adversarial anonymization and empirical synthetic-trace evaluation."""
from .core import AdversarialAnonymize, SynthesizeTrace, adversarial_anonymize, synthesize_trace
from .matching import matches, normalize_identity
from .models import (
    DEFAULT_ATTRIBUTES, AbstractProfile, AttackReport, AttemptSummary, GroundTruth,
    Hint, Inference, JSONRequest, JSONResponse, MatterGuess, ModelResponseError,
    Segment, SynthesisResult, SyntheticWorld, Trace, TraceError, WorldRules,
)
from .trace import (
    DEFAULT_TOOL_VALIDATORS, fill_trace, neutralize_trace, profile_trace, trace_from_dict, trace_to_dict,
    trace_to_source_record,
    validate_shape, validate_trace, validate_world,
)

__all__ = [
    "AdversarialAnonymize", "SynthesizeTrace", "adversarial_anonymize", "synthesize_trace",
    "matches", "normalize_identity", "DEFAULT_ATTRIBUTES", "AbstractProfile", "AttackReport",
    "AttemptSummary", "GroundTruth", "Hint", "Inference", "JSONRequest", "JSONResponse",
    "MatterGuess", "ModelResponseError", "Segment", "SynthesisResult", "SyntheticWorld",
    "Trace", "TraceError", "WorldRules", "DEFAULT_TOOL_VALIDATORS", "fill_trace", "neutralize_trace", "profile_trace",
    "trace_from_dict", "trace_to_dict", "trace_to_source_record", "validate_shape", "validate_trace", "validate_world",
]

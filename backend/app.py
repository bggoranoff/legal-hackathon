"""Mock Orizon demo API: original traces, redact/synthetic transforms, reconstruct, verify."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import traces

app = FastAPI(title="Orizon demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModeBody(BaseModel):
    mode: str  # "redact" or "synthetic"


@app.get("/api/traces/original")
def original():
    return traces.original_traces()


@app.post("/api/redact")
def redact():
    return traces.redact_traces()


@app.post("/api/synthetic")
def synthetic():
    return traces.synthetic_traces()


@app.post("/api/reconstruct")
def reconstruct(body: ModeBody):
    key = "redact" if body.mode == "redact" else "synthetic"
    return traces.reconstruct_traces(key)


@app.post("/api/verify")
def verify(body: ModeBody):
    key = "redact" if body.mode == "redact" else "synthetic"
    return {"score": traces.verify_score(key)}

"""Perfetto-compatible timeline recorder.

Produces Chrome Trace Event Format JSON files that can be opened directly
in https://ui.perfetto.dev or ``chrome://tracing``.

Each pipeline stage is recorded with a distinct color so GPU utilization
and idle gaps are immediately visible.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Generator

# Stage name → (category, cname) mapping for Perfetto colors
STAGE_COLORS: Final[dict[str, tuple[str, str]]] = {
    "Stage 1: CPU Preprocess": ("cpu", "rail_response"),        # Blue
    "Stage 2: CPU+GPU Extract": ("cpu_gpu", "rail_animation"),  # Purple
    "Stage 3: GPU Inference": ("gpu", "thread_state_running"),  # Green
    "Stage 4: CPU/IO Postprocess": ("io", "thread_state_iowait"),  # Orange
}

# Thread IDs for each stage (used as Perfetto sub-tracks)
STAGE_TID: Final[dict[str, int]] = {
    "Stage 1: CPU Preprocess": 1,
    "Stage 2: CPU+GPU Extract": 2,
    "Stage 3: GPU Inference": 3,
    "Stage 4: CPU/IO Postprocess": 4,
}


@dataclass(frozen=True)
class TraceEvent:
    """A single Chrome Trace Event Format event."""

    name: str
    cat: str
    ph: str
    ts: float  # microseconds
    pid: int
    tid: int
    cname: str
    dur: float = 0.0  # microseconds, only for "X" (complete) events
    args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
            "cname": self.cname,
        }
        if self.ph == "X":
            d["dur"] = self.dur
        if self.args:
            d["args"] = self.args
        return d


class TraceRecorder:
    """Records pipeline stage durations as Chrome Trace Event Format events.

    Each ``record()`` call captures a complete duration event ("X" phase).
    Events are stored in-memory and collected at the end of a pipeline run.
    """

    def __init__(self, worker_id: int) -> None:
        self._worker_id = worker_id
        self._events: list[TraceEvent] = []
        self._base_ns: int = time.perf_counter_ns()
        self._lock = threading.Lock()

    def _elapsed_us(self) -> float:
        """Microseconds since recorder creation."""
        return (time.perf_counter_ns() - self._base_ns) / 1_000.0

    @contextmanager
    def record(
        self, stage_name: str, batch_idx: int, batch_size: int = 0
    ) -> Generator[None, None, None]:
        """Context manager that records a complete duration event.

        Usage::

            with recorder.record("Stage 3: GPU Inference", batch_idx=5, batch_size=256):
                result = model(tensor)
        """
        cat, cname = STAGE_COLORS.get(stage_name, ("unknown", "generic_work"))
        tid = STAGE_TID.get(stage_name, 0)

        start_us = self._elapsed_us()
        yield
        dur_us = self._elapsed_us() - start_us

        event = TraceEvent(
            name=f"{stage_name} [batch {batch_idx}]",
            cat=cat,
            ph="X",
            ts=start_us,
            dur=dur_us,
            pid=self._worker_id,
            tid=tid,
            cname=cname,
            args={
                "batch_idx": batch_idx,
                "worker_id": self._worker_id,
                "batch_size": batch_size,
            },
        )
        with self._lock:
            self._events.append(event)

    def get_events(self) -> list[dict]:
        """Return all recorded events as Chrome Trace Format dicts."""
        return [e.to_dict() for e in self._events]

    @staticmethod
    def _build_metadata_events(worker_ids: list[int]) -> list[dict]:
        """Generate process_name and thread_name metadata events."""
        metadata: list[dict] = []
        for wid in worker_ids:
            metadata.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": wid,
                    "tid": 0,
                    "args": {"name": f"Worker {wid}"},
                }
            )
            for stage_name, tid in STAGE_TID.items():
                metadata.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": wid,
                        "tid": tid,
                        "args": {"name": stage_name},
                    }
                )
        return metadata

    @staticmethod
    def merge(all_events: list[list[dict]], worker_ids: list[int]) -> dict:
        """Merge events from multiple workers into a single trace dict."""
        merged: list[dict] = TraceRecorder._build_metadata_events(worker_ids)
        for events in all_events:
            merged.extend(events)
        return {"traceEvents": merged}

    @staticmethod
    def save(
        path: str,
        all_events: list[list[dict]],
        worker_ids: list[int],
    ) -> None:
        """Write merged trace events to a JSON file."""
        trace = TraceRecorder.merge(all_events, worker_ids)
        with open(path, "w") as f:
            json.dump(trace, f, separators=(",", ":"))

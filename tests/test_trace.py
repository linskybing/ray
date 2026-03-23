"""Tests for src/trace.py."""

import json
import os
import tempfile
import threading

from src.trace import (
    STAGE_COLORS,
    STAGE_TID,
    TRACE_THREAD_NAMES,
    TraceEvent,
    TraceRecorder,
)


class TestTraceEvent:
    def test_to_dict_complete_event(self):
        event = TraceEvent(
            name="test",
            cat="gpu",
            ph="X",
            ts=1000.0,
            dur=500.0,
            pid=0,
            tid=3,
            cname="thread_state_running",
        )
        d = event.to_dict()
        assert d["name"] == "test"
        assert d["ph"] == "X"
        assert d["dur"] == 500.0
        assert d["cname"] == "thread_state_running"

    def test_frozen(self):
        event = TraceEvent(
            name="x", cat="y", ph="X", ts=0, pid=0, tid=0, cname="good"
        )
        try:
            event.name = "z"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestTraceRecorder:
    def test_record_captures_event(self):
        recorder = TraceRecorder(worker_id=0)
        with recorder.record("Stage 3: GPU Inference", batch_idx=2):
            pass  # simulate work
        events = recorder.get_events()
        assert len(events) == 1
        assert events[0]["cat"] == "gpu"
        assert events[0]["cname"] == "thread_state_running"
        assert events[0]["dur"] > 0

    def test_record_with_batch_size(self):
        recorder = TraceRecorder(worker_id=0)
        with recorder.record("Stage 1: CPU Preprocess", batch_idx=0, batch_size=32):
            pass
        events = recorder.get_events()
        assert events[0]["args"]["batch_size"] == 32

    def test_counter_captures_value(self):
        recorder = TraceRecorder(worker_id=0)
        recorder.counter("Queue: q_pre depth", 5, args={"maxsize": 14})
        events = recorder.get_events()
        assert events[0]["ph"] == "C"
        assert events[0]["args"]["value"] == 5
        assert events[0]["args"]["maxsize"] == 14

    def test_multiple_stages_different_colors(self):
        recorder = TraceRecorder(worker_id=1)
        for stage_name in STAGE_COLORS:
            with recorder.record(stage_name, batch_idx=0):
                pass
        events = recorder.get_events()
        assert len(events) == 4
        cnames = {e["cname"] for e in events}
        assert len(cnames) == 4  # all different colors

    def test_stage_tid_mapping(self):
        recorder = TraceRecorder(worker_id=0)
        for stage_name in STAGE_TID:
            with recorder.record(stage_name, batch_idx=0):
                pass
        events = recorder.get_events()
        tids = {e["tid"] for e in events}
        assert tids == {1, 2, 3, 4}

    def test_thread_safety(self):
        recorder = TraceRecorder(worker_id=0)
        barrier = threading.Barrier(4)

        def record_from_thread(stage_name: str) -> None:
            barrier.wait()
            for i in range(10):
                with recorder.record(stage_name, batch_idx=i, batch_size=8):
                    pass

        threads = [
            threading.Thread(target=record_from_thread, args=(name,))
            for name in STAGE_COLORS
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = recorder.get_events()
        assert len(events) == 40  # 4 stages × 10 batches


class TestMergeAndSave:
    def test_merge_includes_metadata(self):
        trace = TraceRecorder.merge([], worker_ids=[0, 1])
        events = trace["traceEvents"]
        metadata = [e for e in events if e["ph"] == "M"]
        expected = 2 * (1 + len(TRACE_THREAD_NAMES))
        assert len(metadata) == expected

    def test_save_produces_valid_json(self):
        recorder = TraceRecorder(worker_id=0)
        with recorder.record("Stage 3: GPU Inference", batch_idx=0):
            pass

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            TraceRecorder.save(path, [recorder.get_events()], worker_ids=[0])
            with open(path) as f:
                trace = json.load(f)
            assert "traceEvents" in trace
            assert len(trace["traceEvents"]) > 0
        finally:
            os.unlink(path)

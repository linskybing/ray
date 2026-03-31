"""PipelineWorker: Ray actor with a multi-thread pipeline for maximum GPU utilization.

All stages run inside a single actor → single pod.  Multiple dedicated threads
connected by ``queue.Queue`` enable true overlap of CPU and GPU work because
NumPy, PyTorch GPU ops, and ``time.sleep`` all release the GIL.

Optimizations over the initial 3-thread version:
- **Multiple S1 producer threads** (default 2) eliminate the CPU preprocessing
  bottleneck, doubling data feed rate to the GPU.
- **Merged S2+S3** removes the S3 function-call overhead (0.3 ms was noise but
  still adds per-batch latency) and keeps everything in one ``inference_mode()``
  block.
- **CUDA-stream double-buffering**: while stream-A computes batch N, stream-B
  pre-transfers batch N+1 to GPU.  H2D transfer (~50 ms) overlaps with
  compute (~400 ms).
- **Pinned memory**: ``pin_memory() + non_blocking=True`` for 2-3x H2D speedup.
- **Multiple S4 consumer threads** (default 2) drain ``q_post`` faster so the
  GPU thread never blocks on a full output queue.
- **Larger queue buffers** absorb producer bursts and prevent GPU starvation.

Thread layout::

    Thread 1a,1b (CPU):  S1(cpu_batch) ─┐
                                          ├──→ q_pre
    Thread 1c    (CPU):  S1(cpu_batch) ─┘
                                          ▼
    Thread 2     (GPU):  accumulate → S2+S3(gpu_batch) → q_post
                                          │
    Thread 3a,3b (CPU):  consume q_post → S4(io_batch) → results  ◄─┘
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import ray
import torch

from .model import INPUT_CHANNELS, INPUT_SIZE, NUM_CLASSES, ModelConfig, load_model
from .stages import (
    stage1_preprocess,
    stage2_extract_and_infer,
    stage2_extract_and_infer_timed,
    stage4_postprocess,
)
from .trace import TraceRecorder

logger = logging.getLogger(__name__)

NUM_CPUS_PER_WORKER: Final[int] = 8
NUM_GPUS_PER_WORKER: Final[int] = 1

# Sentinel object to signal thread completion
_SENTINEL: Final[object] = object()

# Tuning knobs
NUM_S1_PRODUCERS: Final[int] = 2
NUM_S4_CONSUMERS: Final[int] = 3
Q_PRE_EXTRA_SLOTS: Final[int] = 6   # extra buffer beyond minimum
Q_POST_EXTRA_SLOTS: Final[int] = 12
DEFAULT_WARMUP_BATCHES: Final[int] = 4


@dataclass(frozen=True)
class PipelineConfig:
    """Variable per-stage batch sizes for balanced throughput."""

    cpu_batch_size: int = 32
    gpu_batch_size: int = 256
    gpu_sub_batch_size: int = 64
    io_batch_size: int = 64
    pool_batch_size: int = 64


@ray.remote(num_cpus=NUM_CPUS_PER_WORKER, num_gpus=NUM_GPUS_PER_WORKER)
class PipelineWorker:
    """Stateful actor running all pipeline stages in one pod.

    Resource request ``num_cpus=8, num_gpus=1`` ensures the actor is scheduled
    onto a GPU worker pod with enough CPU headroom for preprocessing.  On the
    K8s side each worker pod claims ``nvidia.com/gpu: 100`` (= 1 full physical
    GPU via MPS).

    The optimized pipeline uses N+1+M threads:
    - N S1 producer threads (CPU preprocess) feeding ``q_pre``
    - 1 GPU thread consuming ``q_pre``, running merged S2+S3, producing to ``q_post``
    - M S4 consumer threads (CPU/IO postprocess) draining ``q_post``
    """

    def __init__(self, worker_id: int, config: ModelConfig | None = None) -> None:
        self.worker_id = worker_id
        self.model, self.device = load_model(config)
        self.trace = TraceRecorder(worker_id)
        self.batches_processed: int = 0
        self._gpu_warmed_up = False

        # Pre-create CUDA streams for double-buffering
        if self.device.type == "cuda":
            self._streams = [
                torch.cuda.Stream(device=self.device),
                torch.cuda.Stream(device=self.device),
            ]
        else:
            self._streams = [None, None]

        logger.info(
            "PipelineWorker %d initialized on %s", worker_id, self.device
        )

    def _record_queue_depth(self, queue_name: str, q: queue.Queue) -> None:
        self.trace.counter(queue_name, q.qsize(), args={"maxsize": q.maxsize})

    def _warmup_gpu_path(self, warmup_batch_size: int) -> None:
        if self.device.type != "cuda" or self._gpu_warmed_up:
            return

        logger.info(
            "PipelineWorker %d running %d warmup batches at warmup_batch_size=%d",
            self.worker_id,
            DEFAULT_WARMUP_BATCHES,
            warmup_batch_size,
        )
        warmup_rng = np.random.default_rng(seed=self.worker_id)
        warmup_shape = (warmup_batch_size, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        for warmup_stream in self._streams:
            for _ in range(DEFAULT_WARMUP_BATCHES):
                warmup_images = warmup_rng.random(warmup_shape, dtype=np.float32)
                stage2_extract_and_infer(
                    warmup_images,
                    self.model.features,
                    self.model.classifier,
                    self.device,
                    stream=warmup_stream,
                )
                if warmup_stream is not None:
                    warmup_stream.synchronize()

        torch.cuda.synchronize(self.device)
        self._gpu_warmed_up = True

    def run_pipeline(self, config: PipelineConfig, task_pool: ray.actor.ActorHandle) -> dict:
        """Run the multi-thread pipeline with variable per-stage batch sizes.

        Returns summary dict with throughput metrics.
        """
        cpu_bs = config.cpu_batch_size
        gpu_bs = config.gpu_batch_size
        gpu_sub_bs = config.gpu_sub_batch_size
        io_bs = config.io_batch_size
        pool_bs = config.pool_batch_size
        if gpu_sub_bs <= 0:
            raise ValueError("gpu_sub_batch_size must be > 0")

        # Larger queues absorb bursts from multiple S1 producers
        q_pre: queue.Queue = queue.Queue(
            maxsize=gpu_bs // cpu_bs + Q_PRE_EXTRA_SLOTS
        )
        q_post: queue.Queue = queue.Queue(
            maxsize=gpu_bs // io_bs + Q_POST_EXTRA_SLOTS
        )

        error_box: list[BaseException] = []
        results: list[dict] = []
        results_lock = threading.Lock()

        # Counter for S1 sentinel tracking (one per producer thread)
        s1_sentinel_count = 0
        s1_sentinel_lock = threading.Lock()
        num_producers = NUM_S1_PRODUCERS

        self._warmup_gpu_path(gpu_sub_bs)
        start_time = time.monotonic()
        model_features = self.model.features
        model_classifier = self.model.classifier

        # ── Pre-fill barrier: S1 producers fill q_pre before GPU starts ──
        prefill_event = threading.Event()

        # ── S1 Producer threads (CPU Preprocess) ─────────────────────
        prefill_signaled = False
        prefill_signal_lock = threading.Lock()
        prefill_target = gpu_bs // cpu_bs  # items needed before GPU starts

        def stage1_thread(thread_idx: int) -> None:
            nonlocal s1_sentinel_count, prefill_signaled
            rng = np.random.default_rng(seed=self.worker_id * 1000 + thread_idx)
            bulk_acquire_size = pool_bs * 4
            local_remaining = 0
            try:
                batch_idx = 0
                while True:
                    # Bulk acquire from TaskPool to reduce RPC round-trips
                    if local_remaining <= 0:
                        granted = ray.get(task_pool.acquire_batch.remote(bulk_acquire_size))
                        if granted == 0:
                            break
                        local_remaining = granted

                    consume = min(local_remaining, pool_bs)
                    remaining = consume
                    while remaining > 0:
                        bs = min(cpu_bs, remaining)
                        with self.trace.record(
                            "Stage 1: CPU Preprocess", batch_idx,
                            batch_size=bs,
                        ):
                            images = stage1_preprocess(bs, rng=rng)
                        with self.trace.record(
                            "Wait: q_pre output",
                            batch_idx=batch_idx,
                            batch_size=bs,
                        ):
                            q_pre.put((images, batch_idx))
                        self._record_queue_depth("Queue: q_pre depth", q_pre)
                        remaining -= bs
                        batch_idx += 1
                        # Signal GPU thread once q_pre has enough data
                        with prefill_signal_lock:
                            if not prefill_signaled and q_pre.qsize() >= prefill_target:
                                prefill_signaled = True
                                prefill_event.set()
                    local_remaining -= consume
            except Exception as exc:
                error_box.append(exc)
            finally:
                with s1_sentinel_lock:
                    s1_sentinel_count += 1
                    if s1_sentinel_count >= num_producers:
                        with self.trace.record("Wait: q_pre output"):
                            q_pre.put(_SENTINEL)
                        self._record_queue_depth("Queue: q_pre depth", q_pre)
                # Ensure GPU thread is unblocked even if S1 finishes early
                prefill_event.set()

        # ── GPU Thread (merged Stage 2+3) ────────────────────────────
        def gpu_thread() -> None:
            try:
                # Wait for S1 producers to pre-fill q_pre before starting
                prefill_event.wait()

                accumulated: list[np.ndarray] = []
                accumulated_count = 0
                gpu_batch_idx = 0

                while True:
                    with self.trace.record(
                        "Wait: q_pre input", batch_idx=gpu_batch_idx
                    ):
                        item = q_pre.get()
                    self._record_queue_depth("Queue: q_pre depth", q_pre)
                    if item is _SENTINEL:
                        if accumulated:
                            _run_gpu_stages(
                                accumulated, gpu_batch_idx,
                                q_post, io_bs, gpu_sub_bs,
                                model_features, model_classifier,
                                self._streams,
                            )
                        break

                    images, _ = item
                    accumulated.append(images)
                    accumulated_count += images.shape[0]

                    if accumulated_count >= gpu_bs:
                        _run_gpu_stages(
                            accumulated, gpu_batch_idx,
                            q_post, io_bs, gpu_sub_bs,
                            model_features, model_classifier,
                            self._streams,
                        )
                        accumulated = []
                        accumulated_count = 0
                        gpu_batch_idx += 1
            except Exception as exc:
                error_box.append(exc)
            finally:
                # Send one sentinel per S4 consumer
                for _ in range(NUM_S4_CONSUMERS):
                    q_post.put(_SENTINEL)

        # ── S4 Consumer threads (CPU/IO Postprocess) ─────────────────
        def stage4_thread(consumer_idx: int) -> None:
            try:
                io_batch_idx = 0
                while True:
                    with self.trace.record(
                        "Wait: q_post input", batch_idx=io_batch_idx
                    ):
                        item = q_post.get()
                    self._record_queue_depth("Queue: q_post depth", q_post)
                    if item is _SENTINEL:
                        break
                    logits_cpu, cpu_features = item
                    with self.trace.record(
                        "Stage 4: CPU/IO Postprocess", io_batch_idx,
                        batch_size=logits_cpu.shape[0],
                    ):
                        result = stage4_postprocess(logits_cpu, cpu_features)
                    with results_lock:
                        results.append(result)
                    io_batch_idx += 1
            except Exception as exc:
                error_box.append(exc)

        # ── Helper: run merged S2+S3 on accumulated data ─────────────
        def _run_gpu_stages(
            accumulated: list[np.ndarray],
            gpu_batch_idx: int,
            out_queue: queue.Queue,
            out_chunk_size: int,
            sub_batch_size: int,
            features_model: torch.nn.Module,
            classifier_model: torch.nn.Module,
            streams: list[torch.cuda.Stream | None],
        ) -> None:
            big_batch = np.concatenate(accumulated, axis=0)
            n = big_batch.shape[0]

            def _emit_gpu_timing(
                timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
                reference_ts_us: float,
                batch_idx: int,
                batch_size: int,
            ) -> None:
                for stage_name, (start_event, end_event) in timing_events.items():
                    self.trace.record_gpu_timed(
                        stage_name,
                        start_event,
                        end_event,
                        reference_ts_us=reference_ts_us,
                        batch_idx=batch_idx,
                        batch_size=batch_size,
                    )

            def _enqueue_chunks(
                logits_cpu: torch.Tensor,
                cpu_features: np.ndarray,
                batch_idx: int,
            ) -> None:
                chunk_counter = 0
                for start in range(0, logits_cpu.shape[0], out_chunk_size):
                    end = min(start + out_chunk_size, logits_cpu.shape[0])
                    with self.trace.record(
                        "Wait: q_post output",
                        batch_idx=batch_idx * 1000 + chunk_counter,
                        batch_size=end - start,
                    ):
                        out_queue.put((logits_cpu[start:end], cpu_features[start:end]))
                    self._record_queue_depth("Queue: q_post depth", out_queue)
                    chunk_counter += 1

            with self.trace.record(
                "Stage 2: CPU+GPU Extract", gpu_batch_idx, batch_size=n
            ):
                for sub_idx, start in enumerate(range(0, n, sub_batch_size)):
                    end = min(start + sub_batch_size, n)
                    sub_images = big_batch[start:end]
                    sub_n = end - start
                    sub_batch_idx = gpu_batch_idx * 10_000 + sub_idx
                    launched_at_us = self.trace.timestamp_us()
                    output_buffer = torch.empty(
                        (sub_n, NUM_CLASSES),
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=self.device.type == "cuda",
                    )
                    del output_buffer
                    # Alternate between streams for H2D/compute overlap
                    stream = streams[sub_idx % len(streams)] if streams else None
                    logits_cpu, cpu_features, timing_events = (
                        stage2_extract_and_infer_timed(
                            sub_images,
                            features_model,
                            classifier_model,
                            self.device,
                            stream=stream,
                            trace_recorder=self.trace,
                            batch_idx=sub_batch_idx,
                        )
                    )

                    if timing_events:
                        _emit_gpu_timing(
                            timing_events,
                            launched_at_us,
                            sub_batch_idx,
                            sub_n,
                        )

                    _enqueue_chunks(
                        logits_cpu,
                        cpu_features,
                        sub_batch_idx,
                    )

                # Synchronize all used streams after all sub-batches
                if self.device.type == "cuda":
                    for s in streams:
                        if s is not None:
                            s.synchronize()

        # ── Launch all threads ───────────────────────────────────────
        s1_threads = []
        for i in range(num_producers):
            t = threading.Thread(
                target=stage1_thread,
                args=(i,),
                name=f"s1-cpu-{i}",
                daemon=True,
            )
            s1_threads.append(t)

        gpu_t = threading.Thread(target=gpu_thread, name="gpu", daemon=True)

        s4_threads = []
        for i in range(NUM_S4_CONSUMERS):
            t = threading.Thread(
                target=stage4_thread,
                args=(i,),
                name=f"s4-io-{i}",
                daemon=True,
            )
            s4_threads.append(t)

        for t in s1_threads:
            t.start()
        gpu_t.start()
        for t in s4_threads:
            t.start()

        for t in s1_threads:
            t.join()
        gpu_t.join()
        for t in s4_threads:
            t.join()

        if error_box:
            raise error_box[0]

        elapsed = time.monotonic() - start_time
        total = sum(r["num_samples"] for r in results)
        self.batches_processed += 1

        return {
            "worker_id": self.worker_id,
            "total_samples": total,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_samples_per_sec": round(
                total / max(elapsed, 1e-9), 1
            ),
            "config": {
                "cpu_batch_size": cpu_bs,
                "gpu_batch_size": gpu_bs,
                    "gpu_sub_batch_size": gpu_sub_bs,
                "io_batch_size": io_bs,
                "pool_batch_size": pool_bs,
            },
        }

    def get_trace_events(self) -> list[dict]:
        """Return recorded trace events for collection by the orchestrator."""
        return self.trace.get_events()

    def get_stats(self) -> dict:
        """Return cumulative stats for this worker."""
        return {
            "worker_id": self.worker_id,
            "total_batches_processed": self.batches_processed,
            "device": str(self.device),
            "gpu_warmed_up": self._gpu_warmed_up,
        }

    def health_check(self) -> bool:
        """Liveness probe — confirms model is loaded and actor is responsive."""
        return True


def _split_work(total: int, num_parts: int) -> list[int]:
    """Split *total* into *num_parts* as-equal-as-possible chunks."""
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if i < remainder else 0) for i in range(num_parts)]

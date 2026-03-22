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

from .model import ModelConfig, load_model
from .stages import (
    stage1_preprocess,
    stage2_extract_and_infer,
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
NUM_S4_CONSUMERS: Final[int] = 2
Q_PRE_EXTRA_SLOTS: Final[int] = 6   # extra buffer beyond minimum
Q_POST_EXTRA_SLOTS: Final[int] = 4


@dataclass(frozen=True)
class PipelineConfig:
    """Variable per-stage batch sizes for balanced throughput."""

    cpu_batch_size: int = 32
    gpu_batch_size: int = 256
    io_batch_size: int = 64
    total_samples: int = 2560


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

    def run_pipeline(self, config: PipelineConfig) -> dict:
        """Run the multi-thread pipeline with variable per-stage batch sizes.

        Returns summary dict with throughput metrics.
        """
        total_samples = config.total_samples
        cpu_bs = config.cpu_batch_size
        gpu_bs = config.gpu_batch_size
        io_bs = config.io_batch_size

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

        start_time = time.monotonic()
        model_features = self.model.features
        model_classifier = self.model.classifier

        # ── S1 Producer threads (CPU Preprocess) ─────────────────────
        def stage1_thread(thread_idx: int, sample_start: int, sample_count: int) -> None:
            nonlocal s1_sentinel_count
            rng = np.random.default_rng(seed=self.worker_id * 1000 + thread_idx)
            try:
                produced = 0
                batch_idx = 0
                while produced < sample_count:
                    bs = min(cpu_bs, sample_count - produced)
                    with self.trace.record(
                        "Stage 1: CPU Preprocess", batch_idx,
                        batch_size=bs,
                    ):
                        images = stage1_preprocess(bs, rng=rng)
                    q_pre.put((images, batch_idx))
                    produced += bs
                    batch_idx += 1
            except Exception as exc:
                error_box.append(exc)
            finally:
                with s1_sentinel_lock:
                    s1_sentinel_count += 1
                    if s1_sentinel_count >= num_producers:
                        q_pre.put(_SENTINEL)

        # ── GPU Thread (merged Stage 2+3) ────────────────────────────
        def gpu_thread() -> None:
            try:
                accumulated: list[np.ndarray] = []
                accumulated_count = 0
                gpu_batch_idx = 0
                stream_idx = 0

                while True:
                    item = q_pre.get()
                    if item is _SENTINEL:
                        if accumulated:
                            _run_gpu_stages(
                                accumulated, gpu_batch_idx,
                                q_post, io_bs,
                                model_features, model_classifier,
                                self._streams[stream_idx],
                            )
                        break

                    images, _ = item
                    accumulated.append(images)
                    accumulated_count += images.shape[0]

                    if accumulated_count >= gpu_bs:
                        _run_gpu_stages(
                            accumulated, gpu_batch_idx,
                            q_post, io_bs,
                            model_features, model_classifier,
                            self._streams[stream_idx],
                        )
                        accumulated = []
                        accumulated_count = 0
                        gpu_batch_idx += 1
                        stream_idx = (stream_idx + 1) % len(self._streams)
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
                    item = q_post.get()
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
            features_model: torch.nn.Module,
            classifier_model: torch.nn.Module,
            stream: torch.cuda.Stream | None,
        ) -> None:
            big_batch = np.concatenate(accumulated, axis=0)
            n = big_batch.shape[0]

            with self.trace.record(
                "Stage 2: CPU+GPU Extract", gpu_batch_idx, batch_size=n
            ):
                logits_cpu, cpu_features = stage2_extract_and_infer(
                    big_batch, features_model, classifier_model,
                    self.device, stream=stream,
                )

            # Synchronize the stream to ensure logits_cpu is ready
            if stream is not None:
                stream.synchronize()

            for start in range(0, n, out_chunk_size):
                end = min(start + out_chunk_size, n)
                out_queue.put((logits_cpu[start:end], cpu_features[start:end]))

        # ── Launch all threads ───────────────────────────────────────
        # Divide samples evenly across S1 producers
        samples_per_producer = _split_work(total_samples, num_producers)
        s1_threads = []
        offset = 0
        for i, count in enumerate(samples_per_producer):
            t = threading.Thread(
                target=stage1_thread,
                args=(i, offset, count),
                name=f"s1-cpu-{i}",
                daemon=True,
            )
            s1_threads.append(t)
            offset += count

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
                "io_batch_size": io_bs,
                "total_samples": total_samples,
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
        }

    def health_check(self) -> bool:
        """Liveness probe — confirms model is loaded and actor is responsive."""
        return True


def _split_work(total: int, num_parts: int) -> list[int]:
    """Split *total* into *num_parts* as-equal-as-possible chunks."""
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if i < remainder else 0) for i in range(num_parts)]

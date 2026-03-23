"""Pipeline orchestrator: creates workers, distributes data, collects results.

Entry point for the KubeRay inference pipeline.  Spawning ``PipelineWorker``
actors that exceed available resources triggers the Ray autoscaler → KubeRay
operator to provision new GPU worker pods.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Final

import ray

from .task_pool import TaskPool
from .trace import TraceRecorder
from .worker import PipelineConfig, PipelineWorker

logger = logging.getLogger(__name__)

DEFAULT_NUM_WORKERS: Final[int] = 6
DEFAULT_CPU_BATCH_SIZE: Final[int] = 32
DEFAULT_GPU_BATCH_SIZE: Final[int] = 256
DEFAULT_GPU_SUB_BATCH_SIZE: Final[int] = 64
DEFAULT_IO_BATCH_SIZE: Final[int] = 64
DEFAULT_POOL_BATCH_SIZE: Final[int] = 64
DEFAULT_TOTAL_SAMPLES: Final[int] = 30720
DEFAULT_TRACE_OUTPUT: Final[str] = "/tmp/ray/pipeline_trace.json"


def create_workers(num_workers: int) -> list:
    """Create ``PipelineWorker`` actors — each lands on its own GPU pod."""
    workers = []
    for i in range(num_workers):
        worker = PipelineWorker.remote(worker_id=i)
        workers.append(worker)
        logger.info("Requested PipelineWorker %d", i)
    return workers


def run_distributed_pipeline(
    num_workers: int = DEFAULT_NUM_WORKERS,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    config: PipelineConfig | None = None,
    trace_output: str = DEFAULT_TRACE_OUTPUT,
) -> dict:
    """Launch the full distributed pipeline.

    Each worker processes its batches independently (no cross-worker deps).
    Workers run in parallel across different pods.

    Flow:
        1. Create actors (triggers autoscaling if needed)
        2. Wait for health checks (model loaded)
        3. Launch pipelines in parallel
        4. Collect results + trace events
        5. Save Perfetto timeline
    """
    if config is None:
        config = PipelineConfig()

    start = time.monotonic()
    task_pool = TaskPool.remote(total_samples=total_samples)

    # 1. Create actors
    workers = create_workers(num_workers)

    # 2. Wait for all workers to be ready
    ready_refs = [w.health_check.remote() for w in workers]
    ray.get(ready_refs)
    logger.info("All %d workers ready (models loaded)", num_workers)

    # 3. Launch pipelines in parallel
    pipeline_refs = [
        w.run_pipeline.remote(config=config, task_pool=task_pool)
        for w in workers
    ]

    # 4. Collect results
    all_results = ray.get(pipeline_refs)

    # 5. Collect trace events from all workers
    trace_refs = [w.get_trace_events.remote() for w in workers]
    all_trace_events = ray.get(trace_refs)

    # 6. Save Perfetto timeline
    worker_ids = list(range(num_workers))
    TraceRecorder.save(trace_output, all_trace_events, worker_ids)
    logger.info("Trace saved to %s", trace_output)

    elapsed = time.monotonic() - start
    total_samples = sum(r["total_samples"] for r in all_results)

    summary = {
        "num_workers": num_workers,
        "total_samples": total_samples,
        "wall_clock_seconds": round(elapsed, 3),
        "aggregate_throughput_sps": round(total_samples / max(elapsed, 1e-9), 1),
        "trace_file": trace_output,
        "config": {
            "cpu_batch_size": config.cpu_batch_size,
            "gpu_batch_size": config.gpu_batch_size,
            "gpu_sub_batch_size": config.gpu_sub_batch_size,
            "io_batch_size": config.io_batch_size,
            "pool_batch_size": config.pool_batch_size,
            "global_total_samples": total_samples,
        },
        "per_worker_results": all_results,
    }

    logger.info(
        "Pipeline complete: %d samples in %.1fs (%.1f samples/sec)",
        total_samples,
        elapsed,
        total_samples / max(elapsed, 1e-9),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KubeRay GPU Inference Pipeline"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of PipelineWorker actors (each gets 1 full GPU)",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="Global total samples shared by all workers via TaskPool",
    )
    parser.add_argument(
        "--cpu-batch-size",
        type=int,
        default=DEFAULT_CPU_BATCH_SIZE,
        help="Batch size for Stage 1 CPU preprocess micro-batches",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=DEFAULT_GPU_BATCH_SIZE,
        help="Batch size for Stage 2+3 GPU processing (larger → higher SM util)",
    )
    parser.add_argument(
        "--gpu-sub-batch-size",
        type=int,
        default=DEFAULT_GPU_SUB_BATCH_SIZE,
        help="Sub-batch size for Stage 2 streaming across CUDA streams",
    )
    parser.add_argument(
        "--io-batch-size",
        type=int,
        default=DEFAULT_IO_BATCH_SIZE,
        help="Batch size for Stage 4 CPU/IO postprocess chunks",
    )
    parser.add_argument(
        "--pool-batch-size",
        type=int,
        default=DEFAULT_POOL_BATCH_SIZE,
        help="Pull size per acquire from the global TaskPool",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default=DEFAULT_TRACE_OUTPUT,
        help="Path for the Perfetto trace JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    config = PipelineConfig(
        cpu_batch_size=args.cpu_batch_size,
        gpu_batch_size=args.gpu_batch_size,
        gpu_sub_batch_size=args.gpu_sub_batch_size,
        io_batch_size=args.io_batch_size,
        pool_batch_size=args.pool_batch_size,
    )

    ray.init()
    try:
        result = run_distributed_pipeline(
            num_workers=args.num_workers,
            total_samples=args.total_samples,
            config=config,
            trace_output=args.trace_output,
        )
        print(f"\n{'='*60}")
        print(f"  Workers:        {result['num_workers']}")
        print(f"  Samples:        {result['total_samples']}")
        print(f"  Wall time:      {result['wall_clock_seconds']}s")
        print(f"  Throughput:     {result['aggregate_throughput_sps']} samples/sec")
        print(f"  CPU batch:      {config.cpu_batch_size}")
        print(f"  GPU batch:      {config.gpu_batch_size}")
        print(f"  GPU sub-batch:  {config.gpu_sub_batch_size}")
        print(f"  IO batch:       {config.io_batch_size}")
        print(f"  Pool batch:     {config.pool_batch_size}")
        print(f"  Global samples: {args.total_samples}")
        print(f"  Trace file:     {result['trace_file']}")
        print(f"{'='*60}")
        print(f"Open in Perfetto: https://ui.perfetto.dev")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

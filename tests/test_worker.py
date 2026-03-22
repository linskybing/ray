"""Integration test for src/worker.py.

Runs PipelineWorker in a local Ray cluster with CPU-only mode
(overrides num_gpus=0) to validate the pipeline flow end-to-end.
"""

import ray
import pytest

from src.worker import PipelineConfig, PipelineWorker, _split_work


@pytest.fixture(scope="module", autouse=True)
def ray_context():
    ray.init(num_cpus=4, num_gpus=0)
    yield
    ray.shutdown()


class TestPipelineWorker:
    def test_health_check(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=0,
            config=None,  # falls back to CPU
        )
        assert ray.get(worker.health_check.remote()) is True

    def test_run_pipeline_returns_results(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=0,
        )
        config = PipelineConfig(
            cpu_batch_size=4, gpu_batch_size=8, io_batch_size=4,
            total_samples=16,
        )
        result = ray.get(worker.run_pipeline.remote(config=config))
        assert result["worker_id"] == 0
        assert result["total_samples"] == 16
        assert result["throughput_samples_per_sec"] > 0

    def test_trace_events_collected(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=1,
        )
        config = PipelineConfig(
            cpu_batch_size=2, gpu_batch_size=4, io_batch_size=2,
            total_samples=8,
        )
        ray.get(worker.run_pipeline.remote(config=config))
        events = ray.get(worker.get_trace_events.remote())

        # S1: 8/2=4 micro-batches (split across producers),
        # S2 (merged S2+S3): 8/4=2 gpu batches,
        # S4: 8/2=4 io chunks → 4+2+4=10 events
        assert len(events) == 10
        cats = {e["cat"] for e in events}
        # No separate "gpu" category anymore — S3 merged into S2
        assert cats == {"cpu", "cpu_gpu", "io"}

    def test_remainder_batch(self):
        """Test non-divisible total_samples is handled correctly."""
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=3,
        )
        config = PipelineConfig(
            cpu_batch_size=3, gpu_batch_size=7, io_batch_size=4,
            total_samples=10,
        )
        result = ray.get(worker.run_pipeline.remote(config=config))
        assert result["total_samples"] == 10

    def test_get_stats(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=2,
        )
        config = PipelineConfig(
            cpu_batch_size=2, gpu_batch_size=4, io_batch_size=2,
            total_samples=4,
        )
        ray.get(worker.run_pipeline.remote(config=config))
        stats = ray.get(worker.get_stats.remote())
        assert stats["worker_id"] == 2
        assert stats["total_batches_processed"] == 1


class TestSplitWork:
    def test_even_split(self):
        assert _split_work(10, 2) == [5, 5]

    def test_uneven_split(self):
        result = _split_work(10, 3)
        assert sum(result) == 10
        assert result == [4, 3, 3]

    def test_single_part(self):
        assert _split_work(7, 1) == [7]

    def test_more_parts_than_total(self):
        result = _split_work(2, 5)
        assert sum(result) == 2
        assert result == [1, 1, 0, 0, 0]

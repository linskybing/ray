"""Integration test for src/worker.py.

Runs PipelineWorker in a local Ray cluster with CPU-only mode
(overrides num_gpus=0) to validate the pipeline flow end-to-end.
"""

import ray
import pytest

from src.task_pool import TaskPool
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
        task_pool = TaskPool.remote(total_samples=16)
        config = PipelineConfig(
            cpu_batch_size=4, gpu_batch_size=8, io_batch_size=4,
            pool_batch_size=6,
        )
        result = ray.get(
            worker.run_pipeline.remote(config=config, task_pool=task_pool)
        )
        assert result["worker_id"] == 0
        assert result["total_samples"] == 16
        assert result["throughput_samples_per_sec"] > 0

    def test_trace_events_collected(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=1,
        )
        task_pool = TaskPool.remote(total_samples=8)
        config = PipelineConfig(
            cpu_batch_size=2, gpu_batch_size=4, io_batch_size=2,
            pool_batch_size=4,
        )
        ray.get(worker.run_pipeline.remote(config=config, task_pool=task_pool))
        events = ray.get(worker.get_trace_events.remote())
        names = {e["name"].split(" [batch ")[0] for e in events}
        cats = {e["cat"] for e in events}

        assert "Stage 1: CPU Preprocess" in names
        assert "Stage 2: CPU+GPU Extract" in names
        assert "Stage 2a: H2D Transfer" in names
        assert "Stage 2b: Backbone" in names
        assert "Stage 3: Classifier" in names
        assert "Stage 2c: D2H Transfer" in names
        assert "Stage 2d: CPU Stats" in names
        assert "Stage 4: CPU/IO Postprocess" in names
        assert "Queue: q_pre depth" in names
        assert "Queue: q_post depth" in names
        assert "Wait: q_pre input" in names
        assert "Wait: q_pre output" in names
        assert "Wait: q_post input" in names
        assert "Wait: q_post output" in names
        assert {"cpu", "cpu_gpu", "gpu", "io", "queue", "transfer", "wait"} <= cats

    def test_remainder_batch(self):
        """Test non-divisible total_samples is handled correctly."""
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=3,
        )
        task_pool = TaskPool.remote(total_samples=10)
        config = PipelineConfig(
            cpu_batch_size=3, gpu_batch_size=7, io_batch_size=4,
            pool_batch_size=6,
        )
        result = ray.get(
            worker.run_pipeline.remote(config=config, task_pool=task_pool)
        )
        assert result["total_samples"] == 10

    def test_get_stats(self):
        worker = PipelineWorker.options(num_gpus=0, num_cpus=1).remote(
            worker_id=2,
        )
        task_pool = TaskPool.remote(total_samples=4)
        config = PipelineConfig(
            cpu_batch_size=2, gpu_batch_size=4, io_batch_size=2,
            pool_batch_size=2,
        )
        ray.get(worker.run_pipeline.remote(config=config, task_pool=task_pool))
        stats = ray.get(worker.get_stats.remote())
        assert stats["worker_id"] == 2
        assert stats["total_batches_processed"] == 1
        assert stats["gpu_warmed_up"] is False


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

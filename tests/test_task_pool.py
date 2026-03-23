"""Tests for src/task_pool.py."""

import threading

import ray
import pytest

from src.task_pool import TaskPool


@pytest.fixture(scope="module", autouse=True)
def ray_context():
    ray.init(num_cpus=4, num_gpus=0)
    yield
    ray.shutdown()


def test_static_pool_exhaustion() -> None:
    pool = TaskPool.remote(total_samples=10)
    assert ray.get(pool.acquire_batch.remote(4)) == 4
    assert ray.get(pool.acquire_batch.remote(4)) == 4
    assert ray.get(pool.acquire_batch.remote(4)) == 2
    assert ray.get(pool.acquire_batch.remote(1)) == 0


def test_acquire_tail_batch() -> None:
    pool = TaskPool.remote(total_samples=3)
    assert ray.get(pool.acquire_batch.remote(10)) == 3
    assert ray.get(pool.acquire_batch.remote(1)) == 0


def test_dynamic_submit() -> None:
    pool = TaskPool.remote(total_samples=0)
    assert ray.get(pool.acquire_batch.remote(1)) == 0
    assert ray.get(pool.submit_work.remote(7)) is True
    assert ray.get(pool.acquire_batch.remote(4)) == 4
    assert ray.get(pool.acquire_batch.remote(4)) == 3


def test_shutdown_blocks_submit() -> None:
    pool = TaskPool.remote(total_samples=2)
    ray.get(pool.shutdown.remote())
    assert ray.get(pool.submit_work.remote(1)) is False
    assert ray.get(pool.acquire_batch.remote(1)) == 0


def test_get_status_counters() -> None:
    pool = TaskPool.remote(total_samples=5)
    assert ray.get(pool.acquire_batch.remote(2)) == 2
    assert ray.get(pool.submit_work.remote(3)) is True
    assert ray.get(pool.acquire_batch.remote(10)) == 6

    status = ray.get(pool.get_status.remote())
    assert status["total_submitted"] == 8
    assert status["total_acquired"] == 8
    assert status["remaining"] == 0
    assert status["is_shutdown"] is False


def test_concurrent_acquire_total_matches() -> None:
    pool = TaskPool.remote(total_samples=100)

    # Drive concurrent client calls from threads; actor serialization guarantees
    # exact accounting under contention.
    acquired_values: list[int] = []
    lock = threading.Lock()

    def grab_many() -> None:
        local_total = 0
        while True:
            got = ray.get(pool.acquire_batch.remote(3))
            if got == 0:
                break
            local_total += got
        with lock:
            acquired_values.append(local_total)

    threads = [threading.Thread(target=grab_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(acquired_values) == 100

    status = ray.get(pool.get_status.remote())
    assert status["remaining"] == 0


def test_invalid_arguments_raise() -> None:
    pool = TaskPool.remote(total_samples=1)
    with pytest.raises(ray.exceptions.RayTaskError, match="batch_size must be > 0"):
        ray.get(pool.acquire_batch.remote(0))
    with pytest.raises(ray.exceptions.RayTaskError, match="num_samples must be > 0"):
        ray.get(pool.submit_work.remote(0))

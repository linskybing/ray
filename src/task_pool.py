"""Global task pool for dynamic work distribution across workers.

All workers pull work from this actor in fixed-size units. This enables
natural load balancing: faster workers drain more tasks, slower workers do less.
"""

from __future__ import annotations

from dataclasses import dataclass

import ray


@dataclass(frozen=True)
class TaskPoolConfig:
    """Configuration for pool behavior.

    total_samples defines static initial capacity. max_pool_size can be used to
    cap dynamic growth (None means unbounded).
    """

    total_samples: int
    pool_batch_size: int = 64
    max_pool_size: int | None = None


@ray.remote(num_cpus=1)
class TaskPool:
    """Centralized work allocator for all workers.

    This actor serializes acquire/submit operations, guaranteeing consistent
    accounting without external locks.
    """

    def __init__(self, total_samples: int, max_pool_size: int | None = None) -> None:
        if total_samples < 0:
            raise ValueError("total_samples must be >= 0")
        if max_pool_size is not None and max_pool_size < total_samples:
            raise ValueError("max_pool_size must be >= total_samples")

        self._remaining = total_samples
        self._max_pool_size = max_pool_size
        self._is_shutdown = False

        self._total_submitted = total_samples
        self._total_acquired = 0

    def acquire_batch(self, batch_size: int) -> int:
        """Acquire up to batch_size samples, returning the granted amount.

        Returns 0 when the pool is exhausted or shut down.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._is_shutdown or self._remaining <= 0:
            return 0

        granted = min(batch_size, self._remaining)
        self._remaining -= granted
        self._total_acquired += granted
        return granted

    def submit_work(self, num_samples: int) -> bool:
        """Submit additional samples to the pool in dynamic mode.

        Returns False if the pool is already shut down.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if self._is_shutdown:
            return False

        if self._max_pool_size is not None:
            next_total = self._remaining + num_samples
            if next_total > self._max_pool_size:
                raise ValueError("submit exceeds max_pool_size")

        self._remaining += num_samples
        self._total_submitted += num_samples
        return True

    def shutdown(self) -> None:
        """Close the pool.

        After shutdown, acquire_batch always returns 0 and submit_work returns
        False.
        """
        self._is_shutdown = True

    def get_status(self) -> dict:
        """Return pool counters and state."""
        return {
            "total_submitted": self._total_submitted,
            "total_acquired": self._total_acquired,
            "remaining": self._remaining,
            "is_shutdown": self._is_shutdown,
        }

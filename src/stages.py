"""Pipeline stages: CPU preprocess → GPU inference → CPU postprocess.

Each function is pure — receives explicit inputs, returns new data, mutates nothing.

Stage workloads:
- ``stage1_preprocess`` — CPU-heavy: RNG image generation, ImageNet normalisation,
  FFT-based frequency filtering, and histogram equalisation.
- ``stage2_extract_and_infer`` — GPU-heavy: merged S2 + S3 with pinned-memory
  H2D transfer and optional CUDA-stream double-buffering.
- ``stage4_postprocess`` — CPU-heavy: softmax, Monte-Carlo uncertainty estimation,
  pairwise embedding distance matrix, and per-sample entropy.
"""

from __future__ import annotations

from typing import Final
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .trace import TraceRecorder

IMAGE_CHANNELS: Final[int] = 3
IMAGE_SIZE: Final[int] = 224

# FFT low-pass filter: keep frequencies within this fraction of max frequency
FFT_CUTOFF_RATIO: Final[float] = 0.75

# Gaussian blur sigma (in pixel-frequency space)
GAUSSIAN_BLUR_SIGMA: Final[float] = 1.5

# Monte-Carlo uncertainty: number of perturbed softmax samples
MC_NUM_SAMPLES: Final[int] = 100
MC_NOISE_SCALE: Final[float] = 0.05

# Top-K nearest neighbours to compute per sample in postprocessing
NMS_TOP_K: Final[int] = 10

# ImageNet normalization constants
IMAGENET_MEAN: Final[np.ndarray] = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
).reshape(1, 3, 1, 1)
IMAGENET_STD: Final[np.ndarray] = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
).reshape(1, 3, 1, 1)


def stage1_preprocess(batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Stage 1 — CPU-bound: generate, normalise, filter, and blur images.

    Pipeline:
      1. Generate random mock images (simulates I/O read from storage).
      2. Apply ImageNet normalisation + noise injection.
      3. FFT low-pass frequency filter (simulates denoise / JPEG artefact
         removal).
      4. Gaussian blur via FFT (simulates anti-aliasing / smoothing pass).

    All heavy operations use ``np.fft`` which releases the GIL.

    Returns:
        ``np.ndarray`` of shape ``(batch_size, 3, 224, 224)``, dtype float32.
    """
    shape = (batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    if rng is not None:
        images = rng.random(shape, dtype=np.float32)
        noise = rng.standard_normal(shape).astype(np.float32) * 0.01
    else:
        images = np.random.rand(*shape).astype(np.float32)
        noise = np.random.randn(*shape).astype(np.float32) * 0.01

    # ── ImageNet normalisation ──
    images = (images - IMAGENET_MEAN) / IMAGENET_STD
    images = images + noise

    # ── FFT low-pass frequency filter ──
    # Transform to frequency domain, zero out high-frequency components,
    # then transform back.  Simulates denoise / anti-aliasing pass.
    freq = np.fft.rfft2(images, axes=(-2, -1))
    rows, cols = IMAGE_SIZE, IMAGE_SIZE // 2 + 1  # rfft2 output shape
    row_cutoff = int(rows * FFT_CUTOFF_RATIO) // 2
    col_cutoff = int(cols * FFT_CUTOFF_RATIO)
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[:row_cutoff, :col_cutoff] = 1.0
    mask[-row_cutoff:, :col_cutoff] = 1.0
    freq = freq * mask
    images = np.fft.irfft2(freq, s=(IMAGE_SIZE, IMAGE_SIZE), axes=(-2, -1)).astype(
        np.float32
    )

    # ── Gaussian blur via FFT ──
    # Second FFT pass: multiply in frequency domain by a Gaussian kernel.
    # Simulates anti-aliasing / edge-aware smoothing.
    y_freq = np.fft.fftfreq(IMAGE_SIZE)[:, np.newaxis]       # (H, 1)
    x_freq = np.fft.rfftfreq(IMAGE_SIZE)[np.newaxis, :]      # (1, W/2+1)
    sigma = GAUSSIAN_BLUR_SIGMA
    gauss_kernel = np.exp(
        -2.0 * (np.pi * sigma) ** 2 * (y_freq ** 2 + x_freq ** 2)
    ).astype(np.float32)                                       # (H, W/2+1)
    freq2 = np.fft.rfft2(images, axes=(-2, -1))
    freq2 = freq2 * gauss_kernel
    images = np.fft.irfft2(freq2, s=(IMAGE_SIZE, IMAGE_SIZE), axes=(-2, -1)).astype(
        np.float32
    )

    return images


def stage2_extract_and_infer(
    images: np.ndarray,
    model_features: nn.Module,
    model_classifier: nn.Module,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    trace_recorder: TraceRecorder | None = None,
    batch_idx: int | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Merged Stage 2+3 — GPU feature extraction + classifier in one shot.

    Transfers images to GPU using pinned memory (``non_blocking=True``),
    runs the conv backbone *and* FC head inside a single
    ``inference_mode()`` block, then transfers logits back to CPU.

    When *stream* is provided the GPU work runs on that stream so the
    caller can overlap H2D / D2H transfers with compute on another stream.

    Returns:
        ``(logits_cpu, cpu_features)`` — logits already on CPU, ready for S4.
    """
    # Pinned-memory H2D transfer (2-3x faster than pageable)
    tensor = torch.from_numpy(images)
    if device.type == "cuda":
        tensor = tensor.pin_memory()

    ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
    with ctx:
        with _trace_ctx(
            trace_recorder,
            "Stage 2a: H2D Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            tensor = tensor.to(device, non_blocking=True)
        with torch.inference_mode():
            with _trace_ctx(
                trace_recorder,
                "Stage 2b: Backbone",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                gpu_features = model_features(tensor)
                gpu_features = torch.flatten(gpu_features, 1)
            with _trace_ctx(
                trace_recorder,
                "Stage 3: Classifier",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                logits = model_classifier(gpu_features)
        with _trace_ctx(
            trace_recorder,
            "Stage 2c: D2H Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            logits_cpu = logits.cpu()

    # CPU auxiliary statistics (runs in parallel with GPU work above
    # when non_blocking transfers are used)
    with _trace_ctx(
        trace_recorder,
        "Stage 2d: CPU Stats",
        batch_idx=batch_idx,
        batch_size=images.shape[0],
    ):
        cpu_features = np.concatenate(
            [
                np.mean(images, axis=(2, 3)),
                np.std(images, axis=(2, 3)),
            ],
            axis=1,
        )

    return logits_cpu, cpu_features


def stage2_extract_and_infer_async(
    images: np.ndarray,
    model_features: nn.Module,
    model_classifier: nn.Module,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    output_buffer: torch.Tensor | None = None,
    trace_recorder: TraceRecorder | None = None,
    batch_idx: int | None = None,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    torch.cuda.Event | None,
    dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
]:
    """Async variant of merged Stage 2+3 for stream pipelining.

    On CUDA with a provided stream, this function enqueues H2D, backbone,
    classifier, and D2H copy (to a CPU pinned buffer) and returns immediately
    without synchronizing. Caller can wait on the returned completion event.
    """
    is_cuda_stream = device.type == "cuda" and stream is not None
    tensor = torch.from_numpy(images)
    if device.type == "cuda":
        tensor = tensor.pin_memory()

    timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    completion_event: torch.cuda.Event | None = None

    ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
    with ctx:
        with _trace_ctx(
            trace_recorder,
            "Stage 2a: H2D Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            h2d_start, h2d_end = _record_stage_events(stream, is_cuda_stream)
            tensor = tensor.to(device, non_blocking=True)
            if is_cuda_stream:
                tensor.record_stream(stream)
            _finalize_stage_events(
                timing_events, "Stage 2a: H2D Transfer", h2d_start, h2d_end, stream
            )

        with torch.inference_mode():
            with _trace_ctx(
                trace_recorder,
                "Stage 2b: Backbone",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                backbone_start, backbone_end = _record_stage_events(stream, is_cuda_stream)
                gpu_features = model_features(tensor)
                gpu_features = torch.flatten(gpu_features, 1)
                if is_cuda_stream:
                    gpu_features.record_stream(stream)
                _finalize_stage_events(
                    timing_events,
                    "Stage 2b: Backbone",
                    backbone_start,
                    backbone_end,
                    stream,
                )
            with _trace_ctx(
                trace_recorder,
                "Stage 3: Classifier",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                classifier_start, classifier_end = _record_stage_events(
                    stream, is_cuda_stream
                )
                logits = model_classifier(gpu_features)
                if is_cuda_stream:
                    logits.record_stream(stream)
                _finalize_stage_events(
                    timing_events,
                    "Stage 3: Classifier",
                    classifier_start,
                    classifier_end,
                    stream,
                )

        with _trace_ctx(
            trace_recorder,
            "Stage 2c: D2H Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            d2h_start, d2h_end = _record_stage_events(stream, is_cuda_stream)
            if output_buffer is None:
                output_buffer = torch.empty(
                    logits.shape,
                    dtype=logits.dtype,
                    device="cpu",
                    pin_memory=device.type == "cuda",
                )
            if device.type == "cuda":
                output_buffer.copy_(logits, non_blocking=True)
                if is_cuda_stream:
                    output_buffer.record_stream(stream)
            else:
                output_buffer = logits.cpu()
            _finalize_stage_events(
                timing_events, "Stage 2c: D2H Transfer", d2h_start, d2h_end, stream
            )
            if is_cuda_stream:
                completion_event = torch.cuda.Event(enable_timing=True)
                completion_event.record(stream)

    with _trace_ctx(
        trace_recorder,
        "Stage 2d: CPU Stats",
        batch_idx=batch_idx,
        batch_size=images.shape[0],
    ):
        cpu_features = np.concatenate(
            [
                np.mean(images, axis=(2, 3)),
                np.std(images, axis=(2, 3)),
            ],
            axis=1,
        )

    return output_buffer, cpu_features, completion_event, timing_events


def stage2_extract_and_infer_timed(
    images: np.ndarray,
    model_features: nn.Module,
    model_classifier: nn.Module,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    trace_recorder: TraceRecorder | None = None,
    batch_idx: int | None = None,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
]:
    """Synchronous Stage 2 helper with CUDA event timing.

    This variant records per-stage CUDA events, synchronizes the provided
    stream before returning, and is intended for conservative sub-batch
    streaming on real GPUs.
    """
    is_cuda_stream = device.type == "cuda" and stream is not None
    tensor = torch.from_numpy(images)
    if device.type == "cuda":
        tensor = tensor.pin_memory()

    timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
    with ctx:
        with _trace_ctx(
            trace_recorder,
            "Stage 2a: H2D Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            h2d_start, h2d_end = _record_stage_events(stream, is_cuda_stream)
            tensor = tensor.to(device, non_blocking=True)
            _finalize_stage_events(
                timing_events, "Stage 2a: H2D Transfer", h2d_start, h2d_end, stream
            )

        with torch.inference_mode():
            with _trace_ctx(
                trace_recorder,
                "Stage 2b: Backbone",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                backbone_start, backbone_end = _record_stage_events(stream, is_cuda_stream)
                gpu_features = model_features(tensor)
                gpu_features = torch.flatten(gpu_features, 1)
                _finalize_stage_events(
                    timing_events,
                    "Stage 2b: Backbone",
                    backbone_start,
                    backbone_end,
                    stream,
                )

            with _trace_ctx(
                trace_recorder,
                "Stage 3: Classifier",
                batch_idx=batch_idx,
                batch_size=images.shape[0],
            ):
                classifier_start, classifier_end = _record_stage_events(
                    stream, is_cuda_stream
                )
                logits = model_classifier(gpu_features)
                _finalize_stage_events(
                    timing_events,
                    "Stage 3: Classifier",
                    classifier_start,
                    classifier_end,
                    stream,
                )

        with _trace_ctx(
            trace_recorder,
            "Stage 2c: D2H Transfer",
            batch_idx=batch_idx,
            batch_size=images.shape[0],
        ):
            d2h_start, d2h_end = _record_stage_events(stream, is_cuda_stream)
            logits_cpu = logits.cpu()
            _finalize_stage_events(
                timing_events, "Stage 2c: D2H Transfer", d2h_start, d2h_end, stream
            )

        if stream is not None:
            stream.synchronize()

    with _trace_ctx(
        trace_recorder,
        "Stage 2d: CPU Stats",
        batch_idx=batch_idx,
        batch_size=images.shape[0],
    ):
        cpu_features = np.concatenate(
            [
                np.mean(images, axis=(2, 3)),
                np.std(images, axis=(2, 3)),
            ],
            axis=1,
        )

    return logits_cpu, cpu_features, timing_events


# Keep legacy wrappers so existing tests still pass -----------------------

def stage2_feature_extract(
    images: np.ndarray,
    model_features: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Legacy S2 wrapper — delegates to the merged implementation."""
    tensor = torch.from_numpy(images).to(device, non_blocking=True)
    with torch.inference_mode():
        gpu_features = model_features(tensor)
        gpu_features = torch.flatten(gpu_features, 1)

    cpu_features = np.concatenate(
        [
            np.mean(images, axis=(2, 3)),
            np.std(images, axis=(2, 3)),
        ],
        axis=1,
    )
    return gpu_features, cpu_features


def stage3_inference(
    gpu_features: torch.Tensor,
    model_classifier: nn.Module,
) -> torch.Tensor:
    """Legacy S3 wrapper — kept for backward compatibility in tests."""
    with torch.inference_mode():
        logits = model_classifier(gpu_features)
    return logits


def stage4_postprocess(
    logits: torch.Tensor,
    cpu_features: np.ndarray,
) -> dict:
    """Stage 4 — CPU-bound: comprehensive post-processing.

    Pipeline:
      1. Softmax → argmax predictions + confidences (standard path).
      2. Monte-Carlo uncertainty estimation — perturb logits with Gaussian
         noise ``MC_NUM_SAMPLES`` times, average the softmax distributions,
         then re-derive predictions.  Simulates MC-dropout ensemble.
      3. Per-sample predictive entropy: ``-Σ p·log(p)``.
      4. Pairwise L2 distance matrix across samples in the batch
         (simulates nearest-neighbour retrieval / NMS).

    Returns:
        Dictionary with predictions, confidences, entropy, distance stats,
        and metadata.
    """
    if logits.is_cuda:
        logits = logits.cpu()

    logits_np = logits.numpy()  # (N, C)

    # ── Monte-Carlo uncertainty estimation ──
    # Generate MC_NUM_SAMPLES perturbed copies, softmax each, average.
    # Simulates MC-dropout ensemble inference — CPU-heavy loop.
    mc_rng = np.random.default_rng(seed=42)
    accumulated = np.zeros_like(logits_np)
    for _ in range(MC_NUM_SAMPLES):
        perturbed = logits_np + mc_rng.normal(
            scale=MC_NOISE_SCALE, size=logits_np.shape
        ).astype(np.float32)
        # Numerically-stable softmax in numpy
        shifted = perturbed - perturbed.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        accumulated += exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    avg_probs = accumulated / MC_NUM_SAMPLES  # (N, C)

    predictions = np.argmax(avg_probs, axis=1)
    confidences = np.max(avg_probs, axis=1)

    # ── Per-sample predictive entropy ──
    log_probs = np.log(avg_probs + 1e-12)
    entropy = -np.sum(avg_probs * log_probs, axis=1)  # (N,)

    # ── Pairwise L2 distance matrix ──
    # Uses (a-b)^2 = a^2 - 2ab + b^2 expansion for efficiency.
    sq_norms = np.sum(avg_probs ** 2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = sq_norms - 2.0 * (avg_probs @ avg_probs.T) + sq_norms.T  # (N, N)
    np.maximum(dist_sq, 0.0, out=dist_sq)  # clamp numerical negatives
    dist_matrix = np.sqrt(dist_sq)  # (N, N)

    # ── Top-K nearest-neighbour ranking (simulates NMS / retrieval) ──
    k = min(NMS_TOP_K, dist_matrix.shape[0])
    top_k_indices = np.argsort(dist_matrix, axis=1)[:, :k]      # (N, K)
    top_k_dists = np.take_along_axis(dist_matrix, top_k_indices, axis=1)
    mean_knn_distance = float(np.mean(top_k_dists))

    return {
        "predictions": predictions.tolist(),
        "confidences": confidences.tolist(),
        "num_samples": len(predictions),
        "auxiliary_feature_dim": cpu_features.shape[1],
        "entropy": entropy.tolist(),
        "mean_pairwise_distance": float(np.mean(dist_matrix)),
        "mean_knn_distance": mean_knn_distance,
    }


# ── helpers ──────────────────────────────────────────────────────────

from contextlib import contextmanager, nullcontext
from typing import Generator


def _null_ctx() -> contextmanager:
    """Return a no-op context manager when no CUDA stream is needed."""
    return nullcontext()


def _trace_ctx(
    trace_recorder: TraceRecorder | None,
    event_name: str,
    batch_idx: int | None,
    batch_size: int,
) -> contextmanager:
    """Return trace recorder context when tracing is enabled."""
    if trace_recorder is None:
        return nullcontext()
    return trace_recorder.record(
        event_name,
        batch_idx=batch_idx,
        batch_size=batch_size,
    )


def _record_stage_events(
    stream: torch.cuda.Stream | None,
    enabled: bool,
) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if not enabled:
        return None, None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    return start, end


def _finalize_stage_events(
    timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
    stage_name: str,
    start: torch.cuda.Event | None,
    end: torch.cuda.Event | None,
    stream: torch.cuda.Stream | None,
) -> None:
    if start is None or end is None:
        return
    end.record(stream)
    timing_events[stage_name] = (start, end)

# KubeRay GPU Inference Pipeline Implementation and Optimization

## 1. Project Objective

Deploy a 4-stage GPU inference pipeline on a Kubernetes cluster using KubeRay with the following requirements:

- All 4 stages (CPU → CPU+GPU → GPU → CPU/IO) execute **within a single Worker Pod**, avoiding inter-pod cold-start overhead
- All 4 stages (CPU → CPU+GPU → GPU → CPU postprocess) execute **within a single Worker Pod**, avoiding inter-pod cold-start overhead
- Each Worker Pod occupies **one complete physical GPU** (NVIDIA MPS quota 100)
- Auto-scale Worker Pods via KubeRay Autoscaler
- Generate Perfetto timeline visualization of GPU utilization and idle gaps
- Maximize GPU SM utilization

## 2. Cluster Environment

| Resource | Specification |
|----------|---------------|
| GPU Nodes | gpu1 / gpu2 / gpu3 |
| CPU per Node | 384 cores |
| RAM per Node | ~96 GB |
| GPU per Node | 4 physical GPUs |
| MPS Configuration | 100 quota per GPU, 400 virtual GPUs/node total |
| Container Registry | Harbor @ `192.168.110.1:30003` |
| Storage | Longhorn |
| Kubernetes Version | v1.35.0 |

## 3. Architecture Design

### Why Ray Actor Instead of Ray Data

Ray Data's `map_batches()` distributes different stages across different workers, violating the requirement that "all stages run in the same pod". Therefore, we chose **Ray Remote Actor**: one Actor = one Pod = one GPU, with all stages executing inside the Actor.

### Model Partitioning

**ResNet-152** (pretrained) is split into two parts:

- `features`: conv1 → bn1 → relu → maxpool → layer1~4 → avgpool (heavy convolution, GPU-intensive)
- `classifier`: Single Linear(2048 → 1000) layer (lightweight FC)

This allows Stage 2 to run the conv backbone and Stage 3 to run the classifier, revealing the true compute distribution on the GPU.

### 4-Stage Pipeline Breakdown

| Stage | Name | Device | Task |
|-------|------|--------|------|
| S1 | CPU Preprocess | CPU | Generate mock images, ImageNet normalization, FFT low-pass filtering, FFT Gaussian blur |
| S2 | CPU+GPU Extract | GPU+CPU | Send images to GPU to run conv backbone, CPU computes channel statistics in parallel |
| S3 | GPU Inference | GPU | FC classifier inference |
| S4 | CPU Postprocess | CPU | Monte Carlo uncertainty softmax, entropy, pairwise distance matrix, Top-K neighbor ranking |

## 4. Kubernetes Deployment Architecture

### Components

- **Namespace**: `ray-pipeline`
- **KubeRay Operator**: v1.5.1 (Helm-installed)
- **RayCluster**: Head node (coordination only, `num-cpus: 0`) + GPU workers (autoscaling)
- **RayJob**: Submit pipeline command, auto-cleanup on completion

### Worker Pod Resource Configuration

```yaml
resources:
  requests:
    cpu: "8"
    memory: "16Gi"
    nvidia.com/gpu: "100"   # MPS quota 100 = 1 complete physical GPU
```

Ray uses `num-gpus: "1"` so scheduler sees 1 GPU. Kubernetes uses `nvidia.com/gpu: "100"` to claim the full GPU.
`/dev/shm` is mounted with 4Gi to ensure sufficient PyTorch DataLoader shared memory.

### Autoscaling

```yaml
replicas: 0         # No workers initially
minReplicas: 0
maxReplicas: 12     # 3 nodes × 4 GPUs = max 12 workers
```

When an Actor requests GPU resources, Ray autoscaler notifies KubeRay operator to add pods. Workers automatically scale down after 120 seconds idle.

## 5. Perfetto Timeline Visualization

Use **Chrome Trace Event Format** to export JSON, which can be opened directly in [Perfetto UI](https://ui.perfetto.dev).

### Color Mapping

| Stage | Category | Perfetto cname | Color |
|-------|----------|------|-------|
| S1 CPU Preprocess | cpu | `rail_response` | Blue |
| S2 CPU+GPU Extract | cpu_gpu | `rail_animation` | Purple |
| S3 GPU Inference | gpu | `thread_state_running` | Green |
| S4 CPU Postprocess | io | `thread_state_iowait` | Orange |

Each Worker is displayed as a separate Process (pid=worker_id) in Perfetto, with all four stages shown on their respective Thread tracks. Opening the trace provides direct visibility into GPU busy times and idle periods.

## 6. Initial Implementation (Async Double-Buffer)

The first version used Python `asyncio` + `run_in_executor` for CPU/GPU overlap:

- Single asyncio event loop managing two buffers
- CPU preprocess and GPU inference attempted to alternate execution
- Fixed batch_size=128 per step

### Initial Performance

```
3 workers, 7680 samples, 67.8s → 113.2 samples/sec
```

### Initial Bottleneck Analysis

From trace analysis of each stage's latency:

| Stage | Avg Latency | Analysis |
|-------|-------------|----------|
| S1 CPU Preprocess | ~710ms | Normal |
| S2 CPU+GPU Extract | ~750ms | ResNet-152 conv backbone (true GPU work) |
| S3 GPU Inference | ~0.8ms | FC layer too lightweight, barely uses GPU |
| S4 CPU/IO Postprocess | ~670ms | 20ms sleep × batch |

Problems:
1. **S3 barely consumes GPU resources** — FC layer computation is minimal
2. **All stages use same batch size** — Unable to optimize for each stage's different characteristics
3. **GIL limitations in asyncio** — CPU-bound and GPU-bound operations in same event loop still block each other
4. **High GPU idle time** — GPU waits for CPU to finish preprocessing before starting next batch

## 7. Optimized Implementation (Global TaskPool + Variable Batch Pipeline)

### Key Changes

Replace asyncio with **native Python threads** plus a **global TaskPool**, connected via `queue.Queue`:

**TaskPool Actor**: Centralized global work allocator. All workers pull work in `pool_batch` units, so faster workers naturally consume more tasks.

**Thread 1a/1b (S1-CPU)**: Continuously pulls from TaskPool, preprocesses in `cpu_batch=32` chunks, and places into `q_pre`

**Thread 2 (S23-GPU)**: Accumulates from `q_pre` until reaching `gpu_batch=256`, then runs S2+S3 in GPU. After completion, moves logits back to CPU (`logits.cpu()`), then splits into `io_batch=64` chunks and places in `q_post`

**Thread 3a/3b (S4-Postprocess)**: Retrieves chunks from `q_post`, executes CPU-heavy postprocess

### Why Python Threading Works Here

While GIL is a common limitation in Python threading, the following operations **release GIL**:

- NumPy numerical operations (S1's normalization, random)
- PyTorch GPU kernel launch + synchronization (S2's conv, S3's FC)
- NumPy Monte Carlo + distance matrix operations (S4 postprocess)

Therefore, all three threads can truly run in parallel: CPU preprocessing continues preparing the next batch while GPU runs inference.

### Variable Batch Size Design

```python
@dataclass(frozen=True)
class PipelineConfig:
    cpu_batch_size: int = 32      # S1 small batches to fill queue quickly
    gpu_batch_size: int = 256     # S2+S3 large batches for higher SM utilization
    io_batch_size: int = 64       # S4 medium batches to balance CPU postprocess latency
    pool_batch_size: int = 64     # global pull unit from TaskPool
```

Design rationale:

- **cpu_batch=32**: S1 processes each batch in ~127ms; 8 batches accumulate to one GPU batch (~1s). Small batches ensure GPU thread doesn't wait too long for first data batch
- **gpu_batch=256**: Larger batch lets GPU SMs be more fully utilized. ResNet-152 conv backbone at batch=256 costs only 1.56ms per sample vs 5.9ms at batch=128, a **2.7x** speedup
- **io_batch=64**: S4 now runs real CPU numeric workload (Monte Carlo uncertainty + pairwise distance ranking). This keeps postprocess non-trivial while still allowing overlap with GPU compute

### CUDA Stream Safety

GPU thread immediately executes `logits.cpu()` after completing `stage3_inference()`, migrating the tensor to CPU memory. This is necessary because CUDA streams cannot be safely accessed across threads. Only after migration is the tensor placed in `q_post` for S4 thread.

### Error Handling

- Each thread captures exceptions with `try/except`, storing in shared `error_box`
- Thread termination signaled via sentinel object (`_SENTINEL`) to downstream queue
- Main thread `join()`s all three threads, then checks `error_box` and raises if any error occurred

### Thread-Safe Tracing

`TraceRecorder` adds `threading.Lock` to prevent race conditions when multiple threads write trace events simultaneously.

## 8. Optimization Results

### Deployment Results (3 workers, 7680 samples/worker)

```
Workers:        3
Samples:        23,040
Wall time:      60.9s
Throughput:     378.4 samples/sec
```

### Performance Comparison

| Metric | Initial (async, bs=128) | Optimized (3-thread, variable bs) | Improvement |
|------|---------------------|-------------------------------|------|
| Per-worker throughput | 113 sps | 137 sps | +21% |
| Steady-state throughput (excluding warmup) | 113 sps | 158 sps | +40% |
| S2 GPU time per sample | 5.9 ms | 1.56 ms | 2.7x |
| S1-S2 overlap | Minimal | ~4s / worker | Significant |

### Trace Analysis (Steady-state)

| Stage | Event Count | Avg Batch Size | Avg Latency |
|-------|-----------|---------------|---------|
| S1 CPU Preprocess | 720 | 32 | 116 ms |
| S2 CPU+GPU Extract | 90 | 256 | 399 ms (steady-state) |
| S3 GPU Inference | 90 | 256 | 0.5 ms |
| S4 CPU Postprocess | 360 | 64 | ~78 ms (local benchmark after workload update) |

First S2 call requires ~9s (CUDA kernel JIT + cuDNN autotuning); subsequent calls stabilize at ~400ms.

## 9. Project Structure

```
ray/
├── src/
│   ├── model.py       # ResNet-152 partitioning (features + classifier)
│   ├── stages.py      # 4 pure function stages
│   ├── worker.py      # PipelineWorker Ray Actor (3-thread pipeline)
│   ├── pipeline.py    # Orchestrator entry point
│   ├── task_pool.py   # Global work allocator actor
│   └── trace.py       # Perfetto timeline recorder
├── tests/
│   ├── test_model.py
│   ├── test_stages.py
│   ├── test_worker.py
│   ├── test_task_pool.py
│   └── test_trace.py
├── deploy/
│   ├── 00-namespace.yaml
│   ├── 01-install-kuberay-operator.sh
│   ├── 02-raycluster.yaml
│   ├── 03-rayjob.yaml
│   ├── deploy.sh
│   └── teardown.sh
├── Dockerfile
└── requirements.txt
```

## 10. Deployment Process

```bash
# One-command deployment
bash deploy/deploy.sh

# Or step-by-step manually
kubectl apply -f deploy/00-namespace.yaml
bash deploy/01-install-kuberay-operator.sh
kubectl apply -f deploy/02-raycluster.yaml
kubectl apply -f deploy/03-rayjob.yaml

# One-command cleanup
bash deploy/teardown.sh
```

## 11. Future Exploration Directions

1. **CUDA Warmup Batch**: Run a dummy batch before the production pipeline for JIT preheating to eliminate the first batch's 9s cold-start
2. **CUDA Graph**: Use `torch.cuda.CUDAGraph` for fixed-shape S2+S3 to reduce kernel launch overhead
3. **Real Data Sources**: Integrate object store (MinIO / S3) or message queue (Kafka) to replace mock data
4. **Dynamic Batch Size**: Automatically adjust cpu_batch and gpu_batch based on queue fill levels
5. **Multi-Model Pipeline**: Run multiple models within the same Actor (e.g., detection → classification → embedding)

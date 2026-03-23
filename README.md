# KubeRay Pipeline Deployment and Testing Report

This project implements a multi-stage GPU inference pipeline using Ray Actor within a single Worker Pod, avoiding inter-pod data transfer latency.

## 1) Current Implementation Focus

### Architecture
- Ray Actor pattern: 1 Actor = 1 Pod = 1 GPU
- All stages executed within a single Actor
- Auto-scaling managed by KubeRay autoscaler

### Pipeline Optimizations
- Global TaskPool actor for dynamic work-stealing across workers
- S1: Multi-producer threads (CPU preprocessing)
- S2 + S3: Merged into single GPU inference path (reduced context switch overhead)
- Pinned memory + non-blocking transfer
- Dual CUDA stream interleaving
- S4: Multi-consumer threads (CPU-heavy post-processing)
- Enlarged queue buffers to reduce GPU wait time
- torch.compile enabled on CUDA device (auto fallback on failure)

### Stage Workload Update (Latest)
- S1 (CPU): random image generation + ImageNet normalization + FFT low-pass filter + FFT Gaussian blur
- S2+S3 (GPU): merged backbone + classifier inference (pinned memory, optional CUDA stream)
- S4 (CPU): Monte Carlo uncertainty estimation, entropy, pairwise distance matrix, Top-K neighbor ranking

This update removes fake `sleep` latency from S4 and replaces it with real numeric workload.

## 2) Key Files
- src/worker.py: Multi-threaded pipeline main flow
- src/stages.py: Stage computations and data transformations
- src/model.py: Model loading and torch.compile integration
- src/pipeline.py: Distributed orchestration
- deploy/02-raycluster.yaml: KubeRay cluster configuration
- deploy/03-rayjob.yaml: RayJob execution parameters

## 3) Actual Deployment Process (Executed)

### A. Build and Push Image
```bash
docker build -t 192.168.110.1:30003/library/ray-pipeline:latest .
docker push 192.168.110.1:30003/library/ray-pipeline:latest
```

### B. Restart Head and Apply Cluster
```bash
kubectl apply -f deploy/02-raycluster.yaml
kubectl -n ray-pipeline delete pod -l ray.io/node-type=head
kubectl -n ray-pipeline wait --for=condition=Ready pod -l ray.io/node-type=head --timeout=300s
```

### C. Resubmit RayJob
```bash
kubectl -n ray-pipeline delete rayjob inference-pipeline-job --ignore-not-found=true
kubectl apply -f deploy/03-rayjob.yaml
```

## 4) Actual Test Results (Current Execution)

### RayJob Status
- jobStatus: SUCCEEDED
- startTime: 2026-03-22T11:35:06Z
- endTime: 2026-03-22T11:37:39Z

### Submitter Program Output Summary
```text
Pipeline complete: 46080 samples in 130.6s (352.8 samples/sec)
Workers:        6
Samples:        46080
Wall time:      130.623s
Throughput:     352.8 samples/sec
Trace file:     /tmp/ray/pipeline_trace.json
```

### Autoscaling Behavior
Observed in submission logs:
- Adding 6 node(s) of type gpu-workers
- Resized to 48 CPUs, 6 GPUs

This confirms that 6 GPU workers were successfully auto-scaled for this execution.

## 5) Test Verification (Code Level)

Local tests completed successfully:
- tests/test_model.py
- tests/test_stages.py
- tests/test_trace.py
- tests/test_worker.py
- tests/test_task_pool.py

Total: 47 tests passed.

## 5.1) Local Stage Timing (Post-Update)

Measured locally after updating `src/stages.py`:

- S1 (`cpu_batch=32`): ~270.0 ms mean
- S4 (`io_batch=64`): ~78.1 ms mean

These timings confirm that all stages now do substantial work; no stage is effectively empty.

## 6) Common Inspection Commands

```bash
# Check RayJob status
kubectl -n ray-pipeline get rayjob inference-pipeline-job

# Check submitter pod logs
kubectl -n ray-pipeline get pods -l job-name=inference-pipeline-job
kubectl -n ray-pipeline logs <submitter-pod-name>

# Check current pods
kubectl -n ray-pipeline get pods -o wide
```

## 7) Trace File

After execution completes, copy trace from head pod and view stage timeline in Perfetto:

```bash
kubectl -n ray-pipeline get pods -l ray.io/node-type=head
kubectl -n ray-pipeline cp <head-pod-name>:/tmp/ray/pipeline_trace.json ./pipeline_trace.json
```

Perfetto: https://ui.perfetto.dev

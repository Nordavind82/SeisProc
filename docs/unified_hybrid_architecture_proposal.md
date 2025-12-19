# Unified Hybrid Architecture for SeisProc
## Ray + Free-Threaded Python + Rust/PyO3 + C++/Metal

**Document Version:** 1.0
**Date:** 2024-12-19
**Status:** Proposal

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Unified Architecture Overview](#2-unified-architecture-overview)
3. [Job Management System](#3-job-management-system)
4. [UI/UX Design](#4-uiux-design)
5. [SEGY I/O Integration](#5-segy-io-integration)
6. [CPU Algorithm Support](#6-cpu-algorithm-support)
7. [Monitoring & Observability](#7-monitoring--observability)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Technical Reference](#9-technical-reference)

---

## 1. Executive Summary

### 1.1 Current State

| Component | Technology | Limitations |
|-----------|------------|-------------|
| UI Framework | PyQt6 | Signal-based, well-architected |
| Parallelism | ProcessPoolExecutor | No cancellation, no pause |
| GPU | C++/Metal via pybind11 | Works but not thread-safe for free-threading |
| CPU Processors | 21 algorithms (Joblib/Numba) | Scattered parallelization |
| SEGY I/O | Python multiprocessing | Slow header parsing |
| Monitoring | Scattered across dialogs | No unified dashboard |

### 1.2 Proposed State

| Component | Technology | Benefits |
|-----------|------------|----------|
| Orchestration | Ray | Fault tolerance, distribution, cancellation |
| Python | Free-threaded (3.14t) | True thread parallelism |
| GPU | C++/Metal (upgraded pybind11) | Thread-safe, same performance |
| CPU | Rust/PyO3 for hot paths + existing Numba | 5-20x faster I/O |
| UI | Unified Job Dashboard | Real-time monitoring |
| Cancellation | Multi-level | < 2 second response |

### 1.3 Key Improvements

- **Fast Cancellation**: < 2 seconds vs unbounded
- **Pause/Resume**: Full checkpoint support
- **Fault Tolerance**: Automatic retry, partial results preserved
- **SEGY I/O**: 5-20x faster with Rust
- **Unified Monitoring**: Single dashboard for all jobs
- **System Resilience**: Heartbeat monitoring, graceful degradation

---

## 2. Unified Architecture Overview

### 2.1 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PyQt6 UI LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ Job Manager │  │  Dashboard  │  │  Seismic    │  │  Settings   │       │
│  │   Dialog    │  │   Widget    │  │   Viewer    │  │   Dialog    │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         └─────────────────┴─────────────────┴─────────────────┘             │
│                                    │                                        │
│                          pyqtSignal (thread-safe)                          │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        JOB ORCHESTRATION LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RAY CLUSTER                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │ Job Scheduler│  │ Object Store │  │   Dashboard  │              │   │
│  │  │  (Driver)    │  │   (Plasma)   │  │   (8265)     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                 │
│              ▼                     ▼                     ▼                  │
│  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐        │
│  │   SEGY Worker     │ │ Processing Worker │ │   Export Worker   │        │
│  │   (Ray Actor)     │ │   (Ray Actor)     │ │   (Ray Actor)     │        │
│  └───────────────────┘ └───────────────────┘ └───────────────────┘        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FREE-THREADED PYTHON 3.14t                               │
│                         PYTHON_GIL=0                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              ThreadPoolExecutor (true parallelism)                   │   │
│  │    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │    │Thread 1 │  │Thread 2 │  │Thread 3 │  │Thread N │              │   │
│  │    └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘              │   │
│  └─────────┼────────────┼────────────┼────────────┼─────────────────────┘   │
└────────────┼────────────┼────────────┼────────────┼─────────────────────────┘
             │            │            │            │
             ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMPUTATION LAYER                                     │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   RUST + PyO3       │  │   C++ + pybind11    │  │   PYTHON + Numba    │ │
│  │   (gil_used=false)  │  │   (mod_gil_not_used)│  │   (JIT compiled)    │ │
│  │   ─────────────────  │  │   ─────────────────  │  │   ─────────────────  │ │
│  │   • SEGY I/O        │  │   • Metal GPU       │  │   • DWT/SWT         │ │
│  │   • Header parsing  │  │   • DWT kernel      │  │   • NMO correction  │ │
│  │   • Sorting         │  │   • STFT kernel     │  │   • Deconvolution   │ │
│  │   • Data validation │  │   • FKK kernel      │  │   • AGC             │ │
│  │   • Rayon parallel  │  │   • Gabor kernel    │  │   • TF-Denoise      │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘ │
│             │                        │                        │             │
│             └────────────────────────┼────────────────────────┘             │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED DATA LAYER                                 │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │   │ Ray Object  │  │    Zarr     │  │   Arrow     │                 │   │
│  │   │   Store     │  │  (chunked)  │  │  (headers)  │                 │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE LAYER                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   APPLE SILICON     │  │    UNIFIED MEMORY   │  │      STORAGE        │ │
│  │   Metal GPU         │  │    (shared CPU/GPU) │  │    NVMe SSD         │ │
│  │   (.metallib)       │  │                     │  │    Zarr chunks      │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Task Type Routing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TASK ROUTER                                         │
│                                                                             │
│   Incoming Task                                                             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    TASK CLASSIFICATION                               │  │
│   │                                                                      │  │
│   │  Is GPU-accelerated?  ─────Yes────▶  Route to Metal C++ Worker     │  │
│   │        │                             (DWT, STFT, FKK, Gabor)        │  │
│   │        No                                                            │  │
│   │        ▼                                                             │  │
│   │  Is I/O-bound?  ───────────Yes────▶  Route to Rust SEGY Worker     │  │
│   │        │                             (Import, Export, Headers)       │  │
│   │        No                                                            │  │
│   │        ▼                                                             │  │
│   │  Is Numba-optimized?  ─────Yes────▶  Route to Python Numba Worker  │  │
│   │        │                             (NMO, Deconv, 3D Denoise)       │  │
│   │        No                                                            │  │
│   │        ▼                                                             │  │
│   │  Default  ─────────────────────────▶  Route to Python CPU Worker   │  │
│   │                                       (Bandpass, AGC, Stacking)      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| UI | PyQt6 | Desktop application interface |
| Orchestration | Ray | Distributed task scheduling, fault tolerance |
| Runtime | Python 3.14t | Free-threaded execution |
| GPU Compute | C++ / Metal / pybind11 | Hardware-accelerated algorithms |
| CPU Compute (Hot) | Rust / PyO3 / Rayon | High-performance I/O and parsing |
| CPU Compute (Existing) | Python / Numba / Joblib | Optimized numerical algorithms |
| Data Storage | Zarr / Arrow / Parquet | Chunked arrays and columnar data |
| Object Sharing | Ray Plasma | Zero-copy data transfer |

---

## 3. Job Management System

### 3.1 Job Lifecycle State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOB STATE MACHINE                                   │
│                                                                             │
│                              ┌─────────┐                                    │
│                              │ CREATED │                                    │
│                              └────┬────┘                                    │
│                                   │ submit()                                │
│                                   ▼                                         │
│                              ┌─────────┐                                    │
│                     ┌────────│ QUEUED  │────────┐                          │
│                     │        └────┬────┘        │                          │
│                     │             │ schedule()  │ cancel()                 │
│                     │             ▼             │                          │
│                     │        ┌─────────┐        │                          │
│          pause() ◀──┼────────│ RUNNING │────────┼──▶ cancel()             │
│                     │        └────┬────┘        │                          │
│                     │             │             │                          │
│                     ▼             │             ▼                          │
│                ┌─────────┐        │        ┌──────────┐                    │
│                │ PAUSED  │        │        │CANCELLING│                    │
│                └────┬────┘        │        └────┬─────┘                    │
│                     │ resume()    │             │                          │
│                     └─────────────┤             │                          │
│                                   │             │                          │
│          ┌────────────────────────┼─────────────┼────────────────┐         │
│          │                        │             │                │         │
│          ▼                        ▼             ▼                ▼         │
│     ┌─────────┐             ┌─────────┐   ┌──────────┐     ┌─────────┐    │
│     │COMPLETED│             │ FAILED  │   │CANCELLED │     │ TIMEOUT │    │
│     └─────────┘             └─────────┘   └──────────┘     └─────────┘    │
│          │                        │             │                │         │
│          └────────────────────────┴─────────────┴────────────────┘         │
│                                   │                                        │
│                                   ▼                                        │
│                              ┌─────────┐                                   │
│                              │ARCHIVED │                                   │
│                              └─────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Job Data Model

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
from uuid import UUID

class JobType(Enum):
    SEGY_IMPORT = "segy_import"
    PROCESSING = "processing"
    SEGY_EXPORT = "segy_export"
    MIGRATION = "migration"
    QC_BATCH = "qc_batch"
    WORKFLOW = "workflow"

class JobState(Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ARCHIVED = "archived"

@dataclass
class Job:
    # Identity
    id: UUID
    name: str
    job_type: JobType

    # Configuration
    config: Dict
    priority: int  # 1-10, higher = more urgent

    # State
    state: JobState
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Progress
    progress: 'JobProgress'

    # Resources
    resources: 'ResourceRequirements'

    # Relationships
    parent_job_id: Optional[UUID]
    child_job_ids: List[UUID]
    dependencies: List[UUID]  # Jobs that must complete first

@dataclass
class JobProgress:
    phase: str  # 'initializing', 'processing', 'finalizing'
    percent: float  # 0-100
    current_item: int
    total_items: int

    # Per-worker progress
    worker_progress: Dict[str, 'WorkerProgress']

    # Timing
    elapsed_seconds: float
    eta_seconds: Optional[float]
    throughput: float  # items/second

    # Resource usage
    memory_used_mb: float
    gpu_utilization: float
    cpu_utilization: float

@dataclass
class ResourceRequirements:
    min_memory_gb: float
    max_memory_gb: float
    gpu_required: bool
    gpu_memory_gb: Optional[float]
    num_workers: int
    estimated_duration_seconds: Optional[float]

@dataclass
class WorkerProgress:
    worker_id: str
    worker_type: str  # 'metal', 'rust', 'numba', 'python'
    status: str  # 'idle', 'active', 'error'
    current_task: Optional[str]
    progress_percent: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: Optional[float]
```

### 3.3 Job Queue Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOB QUEUE SYSTEM                                    │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      PRIORITY QUEUES                                   │ │
│  │                                                                        │ │
│  │   High Priority (P1)    ┌───┬───┬───┬───┬───┐                        │ │
│  │   Interactive/QC        │ J │ J │ J │   │   │ ──▶ Immediate          │ │
│  │                         └───┴───┴───┴───┴───┘                        │ │
│  │                                                                        │ │
│  │   Normal Priority (P2)  ┌───┬───┬───┬───┬───┬───┬───┬───┐           │ │
│  │   Batch Processing      │ J │ J │ J │ J │ J │ J │   │   │ ──▶ FIFO  │ │
│  │                         └───┴───┴───┴───┴───┴───┴───┴───┘           │ │
│  │                                                                        │ │
│  │   Low Priority (P3)     ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐  │ │
│  │   Background/Export     │ J │ J │ J │ J │ J │ J │ J │ J │   │   │  │ │
│  │                         └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     │                                       │
│                                     ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      SCHEDULER                                         │ │
│  │                                                                        │ │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │ │
│  │   │ Resource Check  │  │ Dependency Check│  │  Priority Sort  │      │ │
│  │   │ (Memory/GPU)    │  │ (Job Graph)     │  │  (P1 > P2 > P3) │      │ │
│  │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │ │
│  │            └────────────────────┼────────────────────┘                │ │
│  │                                 ▼                                      │ │
│  │                    ┌─────────────────────┐                            │ │
│  │                    │   SELECT NEXT JOB   │                            │ │
│  │                    └──────────┬──────────┘                            │ │
│  └───────────────────────────────┼───────────────────────────────────────┘ │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      WORKER POOL                                       │ │
│  │                                                                        │ │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │ │
│  │   │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │            │ │
│  │   │ (GPU)    │  │ (CPU)    │  │ (CPU)    │  │ (I/O)    │            │ │
│  │   │ Metal    │  │ Rust     │  │ Numba    │  │ Rust     │            │ │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Multi-Level Cancellation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-LEVEL CANCELLATION                                 │
│                                                                             │
│   User clicks "Cancel"                                                      │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ LEVEL 1: RAY ORCHESTRATION                                          │  │
│   │                                                                      │  │
│   │   ray.cancel(job_refs, force=False)                                 │  │
│   │   • Sends cancellation signal to all job tasks                      │  │
│   │   • Waits for graceful shutdown (configurable timeout)              │  │
│   │   • If timeout: ray.cancel(job_refs, force=True)                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼ (signal propagates to workers)                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ LEVEL 2: PYTHON WORKERS                                             │  │
│   │                                                                      │  │
│   │   threading.Event() or ray.get_actor_cancellation()                │  │
│   │   • Check event between gather processing                           │  │
│   │   • Raise CancellationError to exit cleanly                        │  │
│   │   • Flush partial results to checkpoint                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼ (passed to native code)                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ LEVEL 3: RUST/C++ LAYER                                             │  │
│   │                                                                      │  │
│   │   std::atomic<bool> cancel_flag                                    │  │
│   │   • Check flag in Rayon parallel iterators                         │  │
│   │   • Early return from computation loops                            │  │
│   │   • Release Metal command buffers                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          │                                                                  │
│          ▼ (GPU cleanup)                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ LEVEL 4: METAL GPU                                                  │  │
│   │                                                                      │  │
│   │   [commandBuffer waitUntilCompleted] or                            │  │
│   │   [commandQueue insertDebugCaptureBoundary]                        │  │
│   │   • Cannot cancel in-flight GPU work                               │  │
│   │   • Wait for current command buffer to finish                      │  │
│   │   • Release subsequent queued buffers                               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ RESPONSE TIMES                                                      │  │
│   │                                                                      │  │
│   │   Level 1 (Ray):     < 100ms  (signal delivery)                    │  │
│   │   Level 2 (Python):  < 1s     (check between gathers)              │  │
│   │   Level 3 (Rust/C++): < 100ms (atomic check in loop)               │  │
│   │   Level 4 (Metal):   < 50ms   (command buffer completion)          │  │
│   │   ─────────────────────────────────────────────────────────────     │  │
│   │   Total worst case:  < 2s     (vs current: unbounded)              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Pause/Resume with Checkpointing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PAUSE/RESUME MECHANISM                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     CHECKPOINT MANAGER                               │  │
│   │                                                                      │  │
│   │   Periodic Checkpoint (every N gathers or T seconds):               │  │
│   │                                                                      │  │
│   │   checkpoint/                                                        │  │
│   │   ├── job_state.json         # Job configuration + progress         │  │
│   │   ├── worker_states/         # Per-worker progress                  │  │
│   │   │   ├── worker_0.json                                             │  │
│   │   │   ├── worker_1.json                                             │  │
│   │   │   └── worker_N.json                                             │  │
│   │   ├── completed_segments.json # Which segments finished             │  │
│   │   └── partial_results/       # Intermediate Zarr data               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     PAUSE FLOW                                       │  │
│   │                                                                      │  │
│   │   1. User clicks "Pause"                                            │  │
│   │   2. Set pause_event (threading.Event)                              │  │
│   │   3. Workers finish current gather                                  │  │
│   │   4. Workers block on pause_event.wait()                           │  │
│   │   5. Coordinator writes checkpoint                                  │  │
│   │   6. UI shows "Paused" state                                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     RESUME FLOW                                      │  │
│   │                                                                      │  │
│   │   1. User clicks "Resume"                                           │  │
│   │   2. Clear pause_event                                              │  │
│   │   3. Workers unblock and continue                                   │  │
│   │   4. UI shows "Running" state                                       │  │
│   │                                                                      │  │
│   │   OR (after app restart):                                           │  │
│   │                                                                      │  │
│   │   1. User selects paused job from history                          │  │
│   │   2. Load checkpoint/job_state.json                                │  │
│   │   3. Skip completed segments                                        │  │
│   │   4. Resume from last checkpoint                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. UI/UX Design

### 4.1 Job Dashboard Widget

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOB DASHBOARD                                   [X] │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ACTIVE JOBS                                              [+ New Job] │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ● SEGY Import: Survey_2024.sgy                    [Pause][Cancel]│ │   │
│  │  │   Status: Importing traces                                      │ │   │
│  │  │   ████████████████████░░░░░░░░░░░░░░  67.2%                    │ │   │
│  │  │   Workers: 6/6 active  |  ETA: 3m 24s  |  Rate: 125K traces/s  │ │   │
│  │  │   Memory: 12.4 GB / 32 GB  |  CPU: 78%                         │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ◐ DWT Processing: Line_100-150                        [Cancel]  │ │   │
│  │  │   Status: Processing gather 45/127 (Line 123)                   │ │   │
│  │  │   ████████████░░░░░░░░░░░░░░░░░░░░░  35.4%                    │ │   │
│  │  │   Workers: 4/4 active  |  ETA: 8m 12s  |  GPU: 92%             │ │   │
│  │  │   Metal: Apple M4 Max  |  Memory: 8.2 GB                       │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ ⏸ SEGY Export: Processed_Output.sgy                  [Resume]  │ │   │
│  │  │   Status: Paused at segment 3/8                                 │ │   │
│  │  │   █████████░░░░░░░░░░░░░░░░░░░░░░░░  28.1%                    │ │   │
│  │  │   Paused for: 12m 34s  |  Can resume anytime                   │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ QUEUED JOBS (3)                                          [Clear All] │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  ○ NMO + Stack: Survey_2024         Priority: Normal    [▲][▼][X]  │   │
│  │  ○ FKK Filter: Survey_2024          Priority: Normal    [▲][▼][X]  │   │
│  │  ○ SEGY Export: Final_Stack.sgy     Priority: Low       [▲][▼][X]  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ RECENT JOBS                                        [Show All][Clear] │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  ✓ SEGY Import: Survey_2023.sgy     Completed   2h ago    [Rerun]  │   │
│  │  ✓ DWT Processing: Line_001-050     Completed   3h ago    [Rerun]  │   │
│  │  ✗ Migration: Bin_A                 Failed      5h ago    [Retry]  │   │
│  │  ⊘ SEGY Export: Test.sgy            Cancelled   6h ago    [Rerun]  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 System Resource Monitor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SYSTEM RESOURCES                               [─][□]│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU Usage                                         GPU Usage                │
│  ┌─────────────────────────────────┐              ┌─────────────────────┐  │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░ 78%       │              │▓▓▓▓▓▓▓▓▓▓▓▓▓░░ 92% │  │
│  │                                 │              │                     │  │
│  │  Cores: 10/12 active           │              │  Metal: Active      │  │
│  │  Threads: 24 running           │              │  Apple M4 Max       │  │
│  └─────────────────────────────────┘              └─────────────────────┘  │
│                                                                             │
│  Memory                                            Disk I/O                 │
│  ┌─────────────────────────────────┐              ┌─────────────────────┐  │
│  │▓▓▓▓▓▓▓▓▓▓░░░░░░░░░ 52%        │              │ Read:  850 MB/s     │  │
│  │                                 │              │ Write: 420 MB/s     │  │
│  │  Used: 16.8 GB / 32 GB         │              │                     │  │
│  │  Available: 15.2 GB            │              │  Queue: 12 ops      │  │
│  └─────────────────────────────────┘              └─────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ WORKER STATUS                                                        │   │
│  ├─────┬──────────┬────────┬────────┬────────┬─────────┬──────────────┤   │
│  │ ID  │ Type     │ Status │ CPU %  │ Mem GB │ Task    │ Progress     │   │
│  ├─────┼──────────┼────────┼────────┼────────┼─────────┼──────────────┤   │
│  │ W0  │ Metal    │ Active │  12%   │  4.2   │ DWT     │ ████░░ 67%  │   │
│  │ W1  │ Rust     │ Active │  95%   │  2.1   │ Import  │ ██████ 89%  │   │
│  │ W2  │ Rust     │ Active │  92%   │  2.0   │ Import  │ █████░ 78%  │   │
│  │ W3  │ Numba    │ Active │  88%   │  3.5   │ NMO     │ ███░░░ 45%  │   │
│  │ W4  │ Numba    │ Idle   │   2%   │  0.5   │ -       │ -           │   │
│  │ W5  │ Python   │ Active │  45%   │  1.2   │ AGC     │ ███████ 92% │   │
│  └─────┴──────────┴────────┴────────┴────────┴─────────┴──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Job Detail View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ JOB DETAILS: DWT Processing - Line_100-150                          [X]    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ OVERVIEW ─────────────────────────────────────────────────────────────┐│
│  │  Job ID: 7f3a2b1c-...                   Created: 2024-12-19 14:23:01  ││
│  │  Type: Parallel Batch Processing        Started: 2024-12-19 14:23:05  ││
│  │  State: ● Running                       Elapsed: 00:12:34             ││
│  │  Priority: Normal                       ETA: 00:08:12                 ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─ PROGRESS ─────────────────────────────────────────────────────────────┐│
│  │                                                                        ││
│  │  Overall:  ████████████████░░░░░░░░░░░░░░░░  52.3%                   ││
│  │                                                                        ││
│  │  Phase: Processing Gathers                                            ││
│  │  Current: Gather 66 of 127 (CDP 1234)                                 ││
│  │  Traces: 1,245,678 / 2,384,521                                        ││
│  │  Throughput: 18,450 traces/sec                                        ││
│  │                                                                        ││
│  │  ┌─ Per-Worker Progress ─────────────────────────────────────────────┐││
│  │  │  W0 (Metal): ████████████████░░░░  80%  Gather 72  GPU: 94%      │││
│  │  │  W1 (Metal): ███████████████░░░░░  75%  Gather 68  GPU: 91%      │││
│  │  │  W2 (Metal): █████████████░░░░░░░  65%  Gather 61  GPU: 88%      │││
│  │  │  W3 (Metal): ██████████░░░░░░░░░░  50%  Gather 54  GPU: 85%      │││
│  │  └──────────────────────────────────────────────────────────────────┘││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─ CONFIGURATION ────────────────────────────────────────────────────────┐│
│  │  Input:     /data/Survey_2024.zarr                                    ││
│  │  Output:    /data/Survey_2024_DWT.zarr                                ││
│  │  Processor: DWTDenoise                                                ││
│  │  Backend:   Metal C++ (Apple M4 Max)                                  ││
│  │  Parameters:                                                           ││
│  │    • Wavelet: db4                                                      ││
│  │    • Levels: 4                                                         ││
│  │    • Threshold: 3.0σ (soft)                                           ││
│  │    • Aperture: 21 traces                                              ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─ LOG ──────────────────────────────────────────────────────────────────┐│
│  │  14:23:05 [INFO]  Job started with 4 Metal workers                    ││
│  │  14:23:06 [INFO]  Loaded ensemble index: 127 gathers                  ││
│  │  14:23:07 [INFO]  Pre-allocated output Zarr: 2.4M traces              ││
│  │  14:25:12 [WARN]  Worker W2 memory pressure, throttling               ││
│  │  14:28:45 [INFO]  Checkpoint saved: 45/127 gathers complete           ││
│  │  14:35:39 [INFO]  Processing gather 66/127 (CDP 1234)                 ││
│  │  ▼ Show more...                                                        ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │   [Pause]   [Cancel]   [View Output]   [Export Log]   [Clone Job]     ││
│  └────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 PyQt6 Signal Architecture

```python
from PyQt6.QtCore import QObject, pyqtSignal

class JobManagerSignals(QObject):
    """Signals emitted by the Job Manager for UI updates."""

    # Job lifecycle
    job_created = pyqtSignal(str)              # job_id
    job_queued = pyqtSignal(str)               # job_id
    job_started = pyqtSignal(str)              # job_id
    job_paused = pyqtSignal(str)               # job_id
    job_resumed = pyqtSignal(str)              # job_id
    job_completed = pyqtSignal(str, bool)      # job_id, success
    job_cancelled = pyqtSignal(str)            # job_id
    job_failed = pyqtSignal(str, str)          # job_id, error_message

    # Progress updates
    progress_updated = pyqtSignal(str, dict)   # job_id, progress_dict
    worker_status_updated = pyqtSignal(str, dict)  # job_id, worker_dict

    # Resource monitoring
    resources_updated = pyqtSignal(dict)       # system_resources
    memory_warning = pyqtSignal(float, float)  # used_gb, available_gb

    # Queue management
    queue_updated = pyqtSignal(list)           # ordered job_ids
    priority_changed = pyqtSignal(str, int)    # job_id, new_priority
```

---

## 5. SEGY I/O Integration

### 5.1 SEGY Operations in Unified Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEGY I/O IN HYBRID ARCHITECTURE                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RAY DRIVER                                   │   │
│  │                                                                      │   │
│  │   ┌──────────────────┐                    ┌──────────────────┐      │   │
│  │   │  SEGYImportJob   │                    │  SEGYExportJob   │      │   │
│  │   │  ──────────────  │                    │  ──────────────  │      │   │
│  │   │  • File analysis │                    │  • Header vector │      │   │
│  │   │  • Partitioning  │                    │  • Partitioning  │      │   │
│  │   │  • Worker spawn  │                    │  • Worker spawn  │      │   │
│  │   │  • Merge results │                    │  • Merge segments│      │   │
│  │   └────────┬─────────┘                    └────────┬─────────┘      │   │
│  └────────────┼──────────────────────────────────────┼──────────────────┘   │
│               │                                      │                      │
│               ▼                                      ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    RUST SEGY WORKERS (PyO3)                         │   │
│  │                    #[pymodule(gil_used = false)]                    │   │
│  │                                                                      │   │
│  │   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │   │
│  │   │ RustImporter   │  │ RustExporter   │  │ RustHeaderProc │       │   │
│  │   │ ────────────── │  │ ────────────── │  │ ────────────── │       │   │
│  │   │ • File read    │  │ • File write   │  │ • Byte parsing │       │   │
│  │   │ • Header parse │  │ • Header write │  │ • Validation   │       │   │
│  │   │ • Zarr write   │  │ • Segment merge│  │ • Computation  │       │   │
│  │   │ • Progress     │  │ • Progress     │  │ • Vectorize    │       │   │
│  │   └────────────────┘  └────────────────┘  └────────────────┘       │   │
│  │                                                                      │   │
│  │   Performance Benefits:                                             │   │
│  │   • Zero-copy buffer protocol with NumPy                           │   │
│  │   • Rayon parallel header parsing                                   │   │
│  │   • Memory-mapped file I/O                                          │   │
│  │   • SIMD-accelerated byte unpacking                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Rust SEGY Module Structure

```
seisproc_rust/
├── Cargo.toml
├── pyproject.toml                 # maturin config
└── src/
    ├── lib.rs                     # PyO3 module definition
    │
    ├── segy/
    │   ├── mod.rs
    │   ├── reader.rs              # SEGY file reading
    │   │   • Memory-mapped file access
    │   │   • IBM float conversion (SIMD)
    │   │   • Parallel trace reading with Rayon
    │   │
    │   ├── writer.rs              # SEGY file writing
    │   │   • Buffered output
    │   │   • Header serialization
    │   │   • Segment merging
    │   │
    │   ├── header.rs              # Header parsing
    │   │   • Big-endian unpacking
    │   │   • Computed headers
    │   │   • Validation
    │   │
    │   └── formats.rs             # Data format handling
    │       • IBM float ↔ IEEE float
    │       • Int16, Int32, Float32
    │
    ├── parallel/
    │   ├── mod.rs
    │   ├── partitioner.rs         # Smart partitioning
    │   ├── progress.rs            # Progress reporting
    │   └── cancellation.rs        # Cancellation tokens
    │
    └── utils/
        ├── mod.rs
        ├── sorting.rs             # Parallel merge sort
        └── validation.rs          # Data validation
```

### 5.3 SEGY Import Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL SEGY IMPORT (RAY + RUST)                        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 1: ANALYSIS (Ray Driver)                                      │  │
│   │                                                                      │  │
│   │   1. Read file metadata (Rust: memory-mapped)                       │  │
│   │   2. Smart partition by ensemble boundaries                         │  │
│   │   3. Pre-create shared Zarr array                                   │  │
│   │   4. Create segment tasks                                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 2: PARALLEL IMPORT (Ray Workers + Rust)                       │  │
│   │                                                                      │  │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│   │   │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker N │           │  │
│   │   │ (Rust)   │  │ (Rust)   │  │ (Rust)   │  │ (Rust)   │           │  │
│   │   │          │  │          │  │          │  │          │           │  │
│   │   │ Segment  │  │ Segment  │  │ Segment  │  │ Segment  │           │  │
│   │   │ 0-25K    │  │ 25K-50K  │  │ 50K-75K  │  │ 75K-100K │           │  │
│   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │  │
│   │        │             │             │             │                  │  │
│   │        └─────────────┴──────┬──────┴─────────────┘                  │  │
│   │                             ▼                                       │  │
│   │                    ┌─────────────────┐                              │  │
│   │                    │   Shared Zarr   │                              │  │
│   │                    │ (direct write)  │                              │  │
│   │                    └─────────────────┘                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 3: MERGE & INDEX (Ray Driver + Rust)                          │  │
│   │                                                                      │  │
│   │   1. Merge segment header parquets (Rust: parallel concat)          │  │
│   │   2. Build ensemble_index.parquet                                   │  │
│   │   3. Save metadata.json                                             │  │
│   │   4. Cleanup temp files                                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   CANCELLATION:                                                            │
│   • Ray: ray.cancel(import_refs)                                          │
│   • Rust: AtomicBool checked every 1000 traces                            │
│   • Cleanup: Remove partial Zarr, temp files                              │
│                                                                             │
│   CHECKPOINT:                                                              │
│   • Save completed segment IDs                                             │
│   • Resume skips completed segments                                        │
│   • Partial results preserved                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 SEGY Export Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL SEGY EXPORT (RAY + RUST)                        │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 1: PREPARATION (Ray Driver)                                   │  │
│   │                                                                      │  │
│   │   1. Vectorize headers to Arrow arrays (Rust)                       │  │
│   │   2. Partition by trace count                                       │  │
│   │   3. Create segment tasks with header slices                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 2: PARALLEL EXPORT (Ray Workers + Rust)                       │  │
│   │                                                                      │  │
│   │   Each Worker:                                                      │  │
│   │   1. Load segment header slice (Arrow)                              │  │
│   │   2. Open Zarr (read-only)                                          │  │
│   │   3. For each trace:                                                │  │
│   │      a. Read from Zarr                                              │  │
│   │      b. Apply mute if configured                                    │  │
│   │      c. Get header from Arrow array (O(1))                         │  │
│   │      d. Write to segment SEG-Y                                      │  │
│   │   4. Report progress                                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ STAGE 3: MERGE SEGMENTS (Rust)                                      │  │
│   │                                                                      │  │
│   │   1. Copy text header from first segment                            │  │
│   │   2. Copy binary header from first segment                          │  │
│   │   3. Concatenate trace data from all segments                       │  │
│   │   4. Delete temp segment files                                      │  │
│   │   5. Verify output file integrity                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   EXPORT TYPES:                                                            │
│   • 'processed': Export processed Zarr directly                           │
│   • 'noise': Calculate (input - processed) on-the-fly                    │
│   • 'both': Export both to separate files                                 │
│                                                                             │
│   MUTE APPLICATION:                                                        │
│   • Top mute: T = |offset| / velocity                                     │
│   • Bottom mute: Same formula                                             │
│   • Cosine taper for smooth transition                                    │
│   • Applied in Rust (SIMD vectorized)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. CPU Algorithm Support

### 6.1 CPU Algorithm Classification

| Tier | Technology | Algorithms | Action |
|------|------------|------------|--------|
| **Tier 1** | Rust + Rayon | Header Parsing, Data Sorting, Validation | Migrate for 5-20x speedup |
| **Tier 2** | Numba JIT | NMO, Deconvolution, Denoise3D | Keep (already 5-20x optimized) |
| **Tier 3** | Joblib Parallel | TFDenoise, STFTDenoise, EMD, SST, OMP, Gabor | Wrap in Ray |
| **Tier 4** | NumPy Vectorized | AGC, Gain, Bandpass, Mute, Stacker, FKFilter | Keep (optimal) |

### 6.2 Complete CPU Processor List

#### Tier 1: Migrate to Rust

| Processor | Current | Complexity | Expected Speedup |
|-----------|---------|------------|------------------|
| Header Parsing | Python | O(n_traces × n_fields) | 10-20x |
| Data Sorting | Python | O(n log n) | 5-10x |
| Data Validation | Python | O(n_traces × n_samples) | 5-10x |
| IBM Float Conversion | Python | O(n_traces × n_samples) | 10-20x |

#### Tier 2: Keep Numba JIT

| Processor | File | Complexity | Current Speedup |
|-----------|------|------------|-----------------|
| NMOProcessor | `nmo_processor.py` | O(n_traces × n_samples) | 10-20x |
| DeconvolutionProcessor | `deconvolution.py` | O(n_traces × window × filter²) | 5-10x |
| Denoise3D | `denoise_3d.py` | O(samples × inlines × xlines × levels) | 5-10x |

#### Tier 3: Wrap in Ray

| Processor | File | Complexity | Parallelization |
|-----------|------|------------|-----------------|
| TFDenoise | `tf_denoise.py` | O(n_traces × n_samples × n_freqs × aperture) | Joblib N-1 cores |
| STFTDenoise | `stft_denoise.py` | O(n_traces × n_samples × n_freqs × aperture) | Joblib N-1 cores |
| StockwellDenoise | `stockwell_denoise.py` | O(n_traces × n_samples × n_freqs × aperture) | Joblib + cache |
| GaborDenoise | `gabor_denoise.py` | O(n_traces × n_samples × n_freqs × aperture) | Joblib N-1 cores |
| SSTDenoise | `sst_denoise.py` | O(n_traces × n_samples × n_freqs × aperture) | Joblib N-1 cores |
| EMDDenoise | `emd_denoise.py` | O(n_traces × n_samples × n_imfs) | Joblib + ensemble |
| OMPDenoise | `omp_denoise.py` | O(n_traces × n_atoms × iterations) | Joblib + Numba |
| DWTDenoise | `dwt_denoise.py` | O(n_traces × n_samples × n_levels) | Joblib + aperture |

#### Tier 4: Keep NumPy Vectorized

| Processor | File | Complexity | Backend |
|-----------|------|------------|---------|
| AGC | `agc.py` | O(n_traces × n_samples) | scipy.ndimage.uniform_filter |
| GainProcessor | `gain_processor.py` | O(n_traces × n_samples) | NumPy scalar multiply |
| BandpassFilter | `bandpass_filter.py` | O(n_traces × n_samples × log n) | scipy.signal.filtfilt |
| MuteProcessor | `mute_processor.py` | O(n_traces × n_samples) | NumPy vectorized |
| CDPStacker | `cdp_stacker.py` | O(n_traces × n_samples) | np.mean/median |
| FKFilter | `fk_filter.py` | O(n_samples × n_traces × log n) | np.fft.fft2 |
| SpectralAnalyzer | `spectral_analyzer.py` | O(n_traces × n_samples × log n) | scipy.fft |

### 6.3 Ray Integration Pattern

```python
import ray
import threading
from typing import List, Dict, Any

@ray.remote(num_cpus=2, memory=4*1024*1024*1024)
class CPUProcessingWorker:
    """Ray actor for CPU-based processing."""

    def __init__(self, processor_config: dict):
        # Reconstruct processor from config
        from processors.base_processor import BaseProcessor
        self.processor = BaseProcessor.from_dict(processor_config)

        # Initialize cancellation
        self.cancel_flag = threading.Event()

    def process_gathers(
        self,
        gather_refs: List[ray.ObjectRef],
        progress_actor: ray.ActorHandle
    ) -> List[ray.ObjectRef]:
        """Process gathers using CPU processor."""
        results = []

        for i, gather_ref in enumerate(gather_refs):
            # Check cancellation
            if self.cancel_flag.is_set():
                raise CancellationError("Processing cancelled")

            # Get data from object store (zero-copy)
            gather_data = ray.get(gather_ref)

            # Process (Joblib/Numba inside)
            result = self.processor.process(gather_data)

            # Put result in object store
            result_ref = ray.put(result)
            results.append(result_ref)

            # Report progress
            progress_actor.update.remote(i + 1, len(gather_refs))

        return results

    def cancel(self):
        """Request cancellation."""
        self.cancel_flag.set()
```

---

## 7. Monitoring & Observability

### 7.1 Metrics Collection Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & OBSERVABILITY                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      METRICS COLLECTION                              │   │
│  │                                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │   │ Ray Metrics  │  │ System Stats │  │ App Metrics  │             │   │
│  │   │ ──────────── │  │ ──────────── │  │ ──────────── │             │   │
│  │   │ • Task count │  │ • CPU usage  │  │ • Job count  │             │   │
│  │   │ • Actor count│  │ • Memory     │  │ • Throughput │             │   │
│  │   │ • Object stor│  │ • Disk I/O   │  │ • Error rate │             │   │
│  │   │ • Scheduling │  │ • GPU util   │  │ • Queue depth│             │   │
│  │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │   │
│  │          └─────────────────┼─────────────────┘                      │   │
│  │                            ▼                                        │   │
│  │                  ┌──────────────────┐                               │   │
│  │                  │  Metrics Store   │                               │   │
│  │                  │  (Time-series)   │                               │   │
│  │                  └────────┬─────────┘                               │   │
│  └───────────────────────────┼──────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┼───────────────┐                             │
│              ▼               ▼               ▼                              │
│  ┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐            │
│  │ PyQt6 Dashboard   │ │ Ray Dashboard │ │ Log Aggregation   │            │
│  │ (embedded widget) │ │ (web :8265)   │ │ (structured logs) │            │
│  └───────────────────┘ └───────────────┘ └───────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Alert System

| Severity | Condition | Action |
|----------|-----------|--------|
| **CRITICAL** | Memory > 90% for 30s | Pause jobs, show dialog |
| **CRITICAL** | Worker crash | Notify, attempt restart |
| **CRITICAL** | Job failed 3x | Stop retries, alert user |
| **CRITICAL** | Disk < 5 GB | Pause exports, alert |
| **WARNING** | Memory > 70% for 2m | Show notification |
| **WARNING** | GPU < 50% during GPU job | Suggest optimization |
| **WARNING** | Job 2x longer than ETA | Show notification |
| **WARNING** | Queue > 10 jobs | Suggest priority review |
| **INFO** | Job started/completed | Log entry |
| **INFO** | Checkpoint saved | Log entry |

### 7.3 Notification Channels

- **System Tray**: Icon badge for job status
- **Toast Popup**: Non-modal notifications
- **Status Bar**: Persistent status message
- **Sound Alert**: Optional audio notification
- **Log Entry**: Always recorded

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goals:**
- Establish Ray infrastructure
- Upgrade pybind11 for free-threading
- Create basic job management

**Tasks:**

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 Ray Setup | Install Ray, create driver/worker pattern | Ray cluster running |
| 1.2 Free-Threading | Upgrade pybind11 v2.13+, add `mod_gil_not_used` | Metal kernels thread-safe |
| 1.3 Job Manager Core | Job data model, state machine, priority queue | Basic job queue |

### Phase 2: UI & SEGY (Weeks 5-8)

**Goals:**
- Job Dashboard UI
- Rust SEGY module
- Pause/Resume functionality

**Tasks:**

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 Job Dashboard | Active jobs, queue, history widgets | Functional dashboard |
| 2.2 Rust SEGY Module | maturin + PyO3, SEGY reader, Rayon parallel | 2-5x faster import |
| 2.3 Pause/Resume | Checkpoint manager, pause event, resume logic | Jobs can pause/resume |

### Phase 3: Processing Integration (Weeks 9-12)

**Goals:**
- All processors running through Ray
- Multi-level cancellation working
- System resource monitoring

**Tasks:**

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 Ray Worker Types | Metal, Rust, Numba, Python workers | Unified processing |
| 3.2 Cancellation Stack | Ray → Python → Rust → Metal | < 2s cancellation |
| 3.3 Resource Monitor | CPU/Memory/GPU gauges, worker table | Real-time monitoring |

### Phase 4: Polish & Optimization (Weeks 13-16)

**Goals:**
- Production-ready stability
- Performance optimization
- Complete documentation

**Tasks:**

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 Rust SEGY Export | Parallel export, segment merge, mute | 2-5x faster export |
| 4.2 Job History | Persistent storage, analytics, retry | Full job tracking |
| 4.3 Testing & Docs | Integration tests, benchmarks, user docs | Production ready |

---

## 9. Technical Reference

### 9.1 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ray | >= 2.9.0 | Distributed computing |
| pybind11 | >= 2.13.0 | C++ Python bindings (free-threading) |
| maturin | >= 1.4.0 | Rust Python packaging |
| PyO3 | >= 0.23.0 | Rust Python bindings |
| rayon | >= 1.10.0 | Rust parallelism |
| PyQt6 | >= 6.6.0 | UI framework |
| zarr | >= 2.16.0 | Chunked array storage |
| pyarrow | >= 14.0.0 | Columnar data |
| numpy | >= 2.0.0 | Numerical arrays |

### 9.2 Configuration Files

#### Cargo.toml (Rust SEGY Module)

```toml
[package]
name = "seisproc_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "seisproc_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
rayon = "1.10"
memmap2 = "0.9"
byteorder = "1.5"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

#### pyproject.toml (Rust Module)

```toml
[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "seisproc_rust"
requires-python = ">=3.11"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### 9.3 pybind11 Free-Threading Update

```cpp
// seismic_metal/src/bindings.cpp

// Before (current):
PYBIND11_MODULE(seismic_metal, m) {
    // ...
}

// After (free-threading support):
PYBIND11_MODULE(seismic_metal, m, py::mod_gil_not_used()) {
    // ...
}
```

### 9.4 Device Manager Thread Safety

```cpp
// seismic_metal/src/device_manager.mm

#include <mutex>

namespace seismic_metal {
    static std::mutex device_mutex;
    static id<MTLDevice> device = nil;

    id<MTLDevice> get_device() {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (!device) {
            device = MTLCreateSystemDefaultDevice();
        }
        return device;
    }
}
```

---

## Summary

This unified architecture provides:

| Feature | Current State | Proposed State |
|---------|--------------|----------------|
| **Orchestration** | ProcessPoolExecutor | Ray (fault-tolerant, distributed) |
| **Cancellation** | Non-functional | < 2 seconds (multi-level) |
| **Pause/Resume** | Not available | Full checkpoint support |
| **GPU Processing** | C++/Metal/pybind11 | Same + free-threading support |
| **CPU Processing** | Joblib/Numba | Same + Ray orchestration |
| **SEGY I/O** | Python (slow) | Rust (5-20x faster) |
| **Job Management** | Per-dialog | Unified dashboard |
| **Monitoring** | External tool | Integrated real-time |
| **System Events** | Vulnerable | Heartbeat + recovery |

The architecture is **incrementally adoptable** - each phase provides immediate value and can be implemented independently.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-19 | Claude | Initial proposal |

---

## References

- [Ray Documentation](https://docs.ray.io/)
- [PyO3 User Guide](https://pyo3.rs/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Python Free-Threading](https://docs.python.org/3/howto/free-threading-python.html)
- [Loky - Robust Executor](https://github.com/joblib/loky)
- [Rayon - Rust Parallelism](https://docs.rs/rayon/)

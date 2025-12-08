# SeisProc Data Storage System - Implementation Tasks

> Generated from comprehensive architecture analysis
> Priority: Critical (P0), High (P1), Medium (P2), Low (P3)

---

## Phase 1: Project & Dataset Management (P0)

### 1.1 DatasetNavigator Class
**File:** `models/dataset_navigator.py`

A centralized manager for multiple loaded datasets with navigation and switching capabilities.

```python
# Target Interface
class DatasetNavigator(QObject):
    dataset_added = pyqtSignal(str, dict)      # dataset_id, metadata
    dataset_removed = pyqtSignal(str)           # dataset_id
    active_dataset_changed = pyqtSignal(str)    # dataset_id

    def add_dataset(self, path: Path, lazy_data: LazySeismicData) -> str
    def remove_dataset(self, dataset_id: str) -> bool
    def get_dataset(self, dataset_id: str) -> Optional[LazySeismicData]
    def set_active_dataset(self, dataset_id: str) -> bool
    def get_active_dataset(self) -> Optional[LazySeismicData]
    def list_datasets(self) -> List[Dict]
    def get_dataset_metadata(self, dataset_id: str) -> Dict
```

**Tasks:**
- [ ] Create `DatasetNavigator` class with Qt signals
- [ ] Implement dataset registry with UUID generation
- [ ] Add active dataset tracking and switching
- [ ] Implement dataset metadata extraction
- [ ] Add dataset comparison utilities
- [ ] Write unit tests for dataset operations

**Acceptance Criteria:**
- Can load multiple SEG-Y files simultaneously
- Switch between datasets without reloading
- Memory-efficient (only active dataset fully cached)

---

### 1.2 AppSettings Dataset Storage Extension
**File:** `models/app_settings.py`

Extend existing AppSettings to persist dataset information across sessions.

**New Settings to Add:**
```python
# Dataset management settings
'loaded_datasets': [],           # List of {id, path, name, loaded_at}
'active_dataset_id': None,       # Currently active dataset UUID
'dataset_cache_limit': 3,        # Max datasets in memory
'auto_load_last_dataset': True,  # Load last dataset on startup

# Session state (NEW)
'session_state': {
    'viewport': {
        'time_min': 0.0,
        'time_max': 1000.0,
        'trace_min': 0.0,
        'trace_max': 100.0,
    },
    'gain': 1.0,
    'colormap': 'seismic',
    'current_gather_id': 0,
}
```

**Tasks:**
- [ ] Add `loaded_datasets` list storage
- [ ] Add `active_dataset_id` tracking
- [ ] Implement `save_session_state()` method
- [ ] Implement `restore_session_state()` method
- [ ] Add dataset path validation on restore
- [ ] Add migration for existing settings
- [ ] Write tests for settings persistence

**Acceptance Criteria:**
- Datasets persist across application restarts
- Session viewport state restored on load
- Invalid/missing dataset paths handled gracefully

---

### 1.3 Project Entity Class
**File:** `models/project.py`

Formal project entity for multi-dataset workflows.

```python
@dataclass
class Project:
    id: UUID
    name: str
    created_at: datetime
    modified_at: datetime
    datasets: List[DatasetReference]
    processing_history: List[ProcessingStep]
    notes: str

@dataclass
class DatasetReference:
    dataset_id: UUID
    source_path: Path
    storage_path: Path  # Zarr directory
    name: str
    added_at: datetime
    metadata: Dict

@dataclass
class ProcessingStep:
    id: UUID
    processor_name: str
    parameters: Dict
    input_dataset_id: UUID
    output_dataset_id: Optional[UUID]
    timestamp: datetime
    checksum: str  # For reproducibility
```

**Tasks:**
- [ ] Create `Project` dataclass
- [ ] Create `DatasetReference` dataclass
- [ ] Create `ProcessingStep` dataclass
- [ ] Implement `ProjectManager` class
- [ ] Add JSON serialization for project files
- [ ] Implement project load/save operations
- [ ] Add project file format versioning

---

## Phase 2: Storage Layer Enhancements (P1)

### 2.1 IDataStore Interface
**File:** `utils/segy_import/data_store_interface.py`

Abstract interface for storage backends (enables future backends).

```python
from abc import ABC, abstractmethod

class IDataStore(ABC):
    @abstractmethod
    def load_dataset(self, path: Path) -> LazySeismicData:
        """Load dataset from storage path."""

    @abstractmethod
    def save_dataset(self, data: SeismicData, path: Path) -> None:
        """Save dataset to storage path."""

    @abstractmethod
    def get_gather(self, dataset_id: str, gather_key: int) -> SeismicData:
        """Retrieve single gather by key."""

    @abstractmethod
    def iter_gathers(self, dataset_id: str) -> Iterator[SeismicData]:
        """Iterate through all gathers."""

    @abstractmethod
    def export(self, dataset_id: str, path: Path,
               format: str, options: Dict) -> None:
        """Export dataset to external format."""
```

**Tasks:**
- [ ] Define `IDataStore` abstract base class
- [ ] Create `ZarrDataStore` implementation (current behavior)
- [ ] Create `ExportOptions` dataclass
- [ ] Add format registry for export handlers
- [ ] Document interface contract

---

### 2.2 Processing Provenance Tracking
**File:** `utils/provenance.py`

Track processing steps for reproducibility.

```python
class ProvenanceTracker:
    def record_step(self, processor: BaseProcessor,
                    input_data: SeismicData,
                    output_data: SeismicData,
                    parameters: Dict) -> ProcessingStep

    def get_history(self, dataset_id: str) -> List[ProcessingStep]
    def compute_checksum(self, data: SeismicData) -> str
    def verify_reproducibility(self, step: ProcessingStep) -> bool
```

**Tasks:**
- [ ] Create `ProvenanceTracker` class
- [ ] Implement checksum computation (MD5/SHA256)
- [ ] Store provenance in dataset metadata
- [ ] Add provenance viewer UI component
- [ ] Export provenance as JSON/YAML

---

### 2.3 Coordinate Reference System Support
**File:** `models/coordinate_system.py`

Add proper CRS/EPSG support for geospatial data.

```python
@dataclass
class CoordinateSystem:
    epsg_code: Optional[int]
    proj_string: Optional[str]
    units: str  # 'meters', 'feet', 'degrees'
    description: str

    @classmethod
    def from_epsg(cls, code: int) -> 'CoordinateSystem'

    def transform_to(self, target_crs: 'CoordinateSystem',
                     coords: np.ndarray) -> np.ndarray
```

**Tasks:**
- [ ] Create `CoordinateSystem` dataclass
- [ ] Add CRS field to dataset metadata
- [ ] Implement EPSG lookup (optional pyproj dependency)
- [ ] Add CRS display in UI
- [ ] Support CRS in SEG-Y export

---

## Phase 3: UI/UX Enhancements (P1)

### 3.1 Session State Persistence
**File:** `models/session_state.py`

Serialize and restore complete session state.

```python
@dataclass
class SessionState:
    viewport: ViewportLimits
    gain: float
    colormap: str
    interpolation: str
    current_dataset_id: Optional[str]
    current_gather_id: int
    sort_keys: List[str]
    panel_layout: Dict

    def to_json(self) -> str
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionState'

class SessionManager:
    def save_session(self, path: Path) -> None
    def load_session(self, path: Path) -> SessionState
    def auto_save(self) -> None  # Called periodically
```

**Tasks:**
- [ ] Create `SessionState` dataclass
- [ ] Implement JSON serialization
- [ ] Add auto-save on application close
- [ ] Add auto-restore on application start
- [ ] Handle missing/corrupted session files
- [ ] Add session reset option in UI

---

### 3.2 Undo/Redo System
**File:** `models/command_history.py`

Command pattern implementation for undo/redo.

```python
class Command(ABC):
    @abstractmethod
    def execute(self) -> None
    @abstractmethod
    def undo(self) -> None
    @abstractmethod
    def description(self) -> str

class CommandHistory:
    def execute(self, command: Command) -> None
    def undo(self) -> Optional[Command]
    def redo(self) -> Optional[Command]
    def can_undo(self) -> bool
    def can_redo(self) -> bool
    def get_history(self) -> List[str]

# Example commands
class ProcessGatherCommand(Command): ...
class ChangeGainCommand(Command): ...
class NavigateGatherCommand(Command): ...
```

**Tasks:**
- [ ] Create `Command` abstract base class
- [ ] Implement `CommandHistory` manager
- [ ] Create `ProcessGatherCommand`
- [ ] Create `ChangeViewportCommand`
- [ ] Add undo/redo keyboard shortcuts (Ctrl+Z, Ctrl+Y)
- [ ] Add undo/redo menu items
- [ ] Display command history in status bar

---

### 3.3 Dataset Switcher UI
**File:** `views/dataset_switcher.py`

UI component for managing multiple datasets.

**Features:**
- Dataset list panel (dockable)
- Quick switch dropdown in toolbar
- Dataset comparison mode (side-by-side)
- Close/unload dataset option
- Dataset properties dialog

**Tasks:**
- [ ] Create `DatasetListWidget` panel
- [ ] Add dataset selector to toolbar
- [ ] Implement drag-and-drop dataset loading
- [ ] Add dataset context menu (close, properties, export)
- [ ] Create `DatasetPropertiesDialog`

---

## Phase 4: Performance Optimizations (P2)

### 4.1 Cache Tuning
**File:** `models/gather_navigator.py`

**Current:** Fixed 5-gather LRU cache
**Target:** Adaptive cache based on available memory

**Tasks:**
- [ ] Make cache size configurable in AppSettings
- [ ] Add memory-based cache limit option
- [ ] Implement adaptive cache sizing based on MemoryMonitor
- [ ] Add cache warm-up on dataset load
- [ ] Add cache statistics to status bar

---

### 4.2 Parallel Prefetching
**File:** `models/gather_navigator.py`

**Current:** Sequential prefetch of +/- 2 gathers
**Target:** Parallel prefetch with ThreadPoolExecutor

**Tasks:**
- [ ] Replace single prefetch thread with ThreadPoolExecutor
- [ ] Add configurable prefetch depth (1-5 gathers)
- [ ] Implement priority-based prefetch queue
- [ ] Cancel outdated prefetch requests on navigation
- [ ] Add prefetch status indicator

---

### 4.3 Lazy Header Loading
**File:** `models/lazy_seismic_data.py`

**Current:** Headers loaded per-gather
**Target:** Column-wise lazy loading for large headers

**Tasks:**
- [ ] Implement column-subset loading for Parquet
- [ ] Cache frequently-accessed header columns
- [ ] Add header statistics without full load
- [ ] Optimize ensemble key extraction

---

## Phase 5: Quality & Robustness (P2)

### 5.1 Error Handling Improvements

**Tasks:**
- [ ] Replace print statements with logging in `data_storage.py`
- [ ] Add structured error types (`StorageError`, `ImportError`)
- [ ] Implement retry logic for I/O operations
- [ ] Add progress recovery for interrupted imports
- [ ] Create error reporting dialog

---

### 5.2 Data Validation

**Tasks:**
- [ ] Validate Zarr array integrity on load
- [ ] Check header-trace count consistency
- [ ] Verify ensemble index completeness
- [ ] Add data repair utilities
- [ ] Implement backup before destructive operations

---

### 5.3 Testing Infrastructure

**Tasks:**
- [ ] Create test fixtures for sample SEG-Y files
- [ ] Add unit tests for DataStorage streaming
- [ ] Add integration tests for full import workflow
- [ ] Add performance benchmarks
- [ ] Set up CI pipeline for tests

---

## Phase 6: Future Enhancements (P3)

### 6.1 SQLite Metadata Backend
Replace JSON metadata with SQLite for queryable storage.

**Tasks:**
- [ ] Design SQLite schema for metadata
- [ ] Migrate metadata.json to SQLite
- [ ] Add query API for metadata
- [ ] Support full-text search on headers

---

### 6.2 Cloud Storage Support
Add support for remote storage backends.

**Tasks:**
- [ ] Abstract file operations behind storage interface
- [ ] Add S3 backend support (boto3)
- [ ] Add Azure Blob support (azure-storage-blob)
- [ ] Implement streaming for cloud sources
- [ ] Add credential management

---

### 6.3 Multi-User Collaboration
Support concurrent access and change tracking.

**Tasks:**
- [ ] Add file locking mechanism
- [ ] Implement change log
- [ ] Add merge/conflict resolution
- [ ] Support read-only mode

---

## Implementation Priority Matrix

| Task | Priority | Effort | Impact | Dependencies |
|------|----------|--------|--------|--------------|
| DatasetNavigator | P0 | Medium | High | None |
| AppSettings Extension | P0 | Low | High | None |
| Project Entity | P0 | Medium | High | DatasetNavigator |
| Session Persistence | P1 | Low | Medium | AppSettings |
| IDataStore Interface | P1 | Medium | Medium | None |
| Provenance Tracking | P1 | Medium | High | Project Entity |
| Undo/Redo System | P1 | High | Medium | None |
| Dataset Switcher UI | P1 | Medium | High | DatasetNavigator |
| Cache Tuning | P2 | Low | Medium | None |
| CRS Support | P2 | Low | Low | None |
| SQLite Backend | P3 | High | Medium | IDataStore |

---

## Quick Start: First 3 Tasks

### Task 1: Create DatasetNavigator (2-3 hours)
```bash
# Create new file
touch models/dataset_navigator.py

# Implement core class with:
# - Dataset registry (dict)
# - Active dataset tracking
# - Qt signals for state changes
# - Integration with GatherNavigator
```

### Task 2: Extend AppSettings (1-2 hours)
```bash
# Modify existing file
# Add loaded_datasets list
# Add session state dict
# Add save/restore methods
```

### Task 3: Wire Up to MainWindow (1-2 hours)
```bash
# Modify main_window.py
# - Create DatasetNavigator instance
# - Connect to GatherNavigator
# - Add dataset menu/toolbar
# - Save state on close
```

---

## File Structure After Implementation

```
models/
├── __init__.py
├── app_settings.py          # Extended with dataset storage
├── dataset_navigator.py     # NEW: Multi-dataset management
├── project.py               # NEW: Project entity
├── session_state.py         # NEW: Session persistence
├── command_history.py       # NEW: Undo/redo
├── coordinate_system.py     # NEW: CRS support
├── gather_navigator.py      # Existing (minor updates)
├── lazy_seismic_data.py     # Existing
├── seismic_data.py          # Existing
├── viewport_state.py        # Existing
└── fk_config.py             # Existing

utils/
├── segy_import/
│   ├── data_store_interface.py  # NEW: Abstract interface
│   ├── data_storage.py          # Existing (implements interface)
│   └── ...
├── provenance.py            # NEW: Processing tracking
└── ...

views/
├── dataset_switcher.py      # NEW: Dataset management UI
└── ...
```

---

## Success Metrics

1. **Dataset Management**
   - Load 5+ datasets without memory issues
   - Switch between datasets in < 100ms
   - Persist dataset list across restarts

2. **Session Persistence**
   - 100% viewport state restored
   - Processing history preserved
   - Graceful handling of missing files

3. **Code Quality**
   - 80%+ test coverage for new code
   - No regressions in existing functionality
   - Documentation for all public APIs

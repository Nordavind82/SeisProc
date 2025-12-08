# Integration Status Report: DatasetNavigator & AppSettings Extensions

> Updated: Integration COMPLETE - App starts successfully

---

## Executive Summary

The new `DatasetNavigator` and extended `AppSettings` classes are now **FULLY INTEGRATED** into the application. All features are connected to the `MainWindow` and startup flow.

### Bug Fix Applied

**Issue:** Segmentation fault on startup when `AppSettings` (a `QObject` subclass) was instantiated before `QApplication` existed.

**Solution:** Refactored `AppSettings` to NOT inherit from `QObject`. Removed Qt signals and use simple logging instead. This makes `AppSettings` safe to import at module load time.

---

## Implementation Status

### Completed Integration Points

| Feature | Status | Location |
|---------|--------|----------|
| `DatasetNavigator` instance | ✅ Done | `MainWindow.__init__` (line 84-88) |
| `AppSettings` instance | ✅ Done | `MainWindow.__init__` (line 65) |
| Session save on close | ✅ Done | `MainWindow.closeEvent` (line 2063) |
| Session restore on start | ✅ Done | `MainWindow.__init__` (line 108-111) |
| Dataset menu items | ✅ Done | `MainWindow._create_menu_bar` (line 214-224) |
| Dataset registration | ✅ Done | `MainWindow._on_segy_imported` (line 529-553) |
| Dataset switching | ✅ Done | `MainWindow._on_active_dataset_changed` (line 1895) |
| Close dataset | ✅ Done | `MainWindow._close_current_dataset` (line 2019) |

### New Methods Added to MainWindow

```python
# Signal connections
_connect_dataset_navigator_signals()

# Dataset handlers
_on_active_dataset_changed(dataset_id)
_on_dataset_added(dataset_id, info_dict)
_on_dataset_removed(dataset_id)
_on_datasets_cleared()
_update_datasets_menu()
_switch_to_dataset(dataset_id)
_close_current_dataset()
_close_all_datasets()

# Session management
closeEvent(event)
_save_current_session()
_restore_last_session()
_apply_pending_session_state()
```

---

## Required Integration Points

### A. Startup Integration

**File: `main_window.py` - `__init__` method**

```python
# ADD these after line 67:
from models.app_settings import get_settings
from models.dataset_navigator import DatasetNavigator

# In __init__:
self.app_settings = get_settings()
self.dataset_navigator = DatasetNavigator(
    max_cached_datasets=self.app_settings.get_dataset_cache_limit()
)

# Connect signals
self.dataset_navigator.active_dataset_changed.connect(self._on_active_dataset_changed)
self.dataset_navigator.dataset_added.connect(self._on_dataset_added)
self.dataset_navigator.dataset_removed.connect(self._on_dataset_removed)
```

### B. Session Restoration

**File: `main_window.py` - end of `__init__`**

```python
# ADD before "Show welcome message":
# Restore session state
if self.app_settings.get_auto_load_last_dataset():
    self._restore_last_session()
```

### C. Session Save on Close

**File: `main_window.py` - ADD new method**

```python
def closeEvent(self, event):
    """Save session state before closing."""
    self._save_current_session()
    event.accept()

def _save_current_session(self):
    """Save current session state to settings."""
    # Save viewport state
    limits = self.viewport_state.limits
    self.app_settings.save_viewport_state(
        limits.time_min, limits.time_max,
        limits.trace_min, limits.trace_max
    )

    # Save display state
    self.app_settings.save_display_state(
        colormap=self.viewport_state.colormap,
        interpolation=self.viewport_state.interpolation
    )

    # Save navigation state
    self.app_settings.save_navigation_state(
        current_gather_id=self.gather_navigator.current_gather_id,
        sort_keys=self.gather_navigator.sort_keys
    )

    # Save active dataset
    if self.dataset_navigator.has_datasets():
        active_id = self.dataset_navigator.get_active_dataset_id()
        self.app_settings.set_active_dataset_id(active_id)

        # Save dataset list
        datasets = self.dataset_navigator.list_datasets()
        self.app_settings.save_loaded_datasets(datasets)
```

### D. Dataset Registration on Import

**File: `main_window.py` - `_on_segy_imported` method**

```python
# ADD after loading data into gather_navigator (line 477-479):

# Register dataset with DatasetNavigator
if isinstance(seismic_data, LazySeismicData):
    storage_path = Path(seismic_data.zarr_path).parent
    dataset_id = self.dataset_navigator.add_dataset(
        source_path=Path(file_path) if file_path else Path(),
        storage_path=storage_path,
        lazy_data=seismic_data,
        name=Path(file_path).stem if file_path else "Imported Data"
    )

    # Save to persistent settings
    info = self.dataset_navigator.get_dataset_info(dataset_id)
    if info:
        self.app_settings.add_loaded_dataset(info.to_dict())
```

### E. Menu Integration

**File: `main_window.py` - `_create_menu_bar` method**

```python
# ADD after "Recent files submenu" (around line 170):

file_menu.addSeparator()

# Dataset management submenu
self.dataset_menu = file_menu.addMenu("&Datasets")
self._update_dataset_menu()

# Close dataset action
close_dataset_action = QAction("Close &Current Dataset", self)
close_dataset_action.triggered.connect(self._close_current_dataset)
file_menu.addAction(close_dataset_action)
```

### F. New Handler Methods Required

```python
def _on_active_dataset_changed(self, dataset_id: str):
    """Handle active dataset change."""
    if not dataset_id:
        return

    lazy_data = self.dataset_navigator.get_dataset(dataset_id)
    if lazy_data is None:
        return

    # Get ensemble index
    ensembles_df = lazy_data._ensemble_index

    # Load into gather navigator
    self.gather_navigator.load_lazy_data(lazy_data, ensembles_df)

    # Display first gather
    self._display_current_gather()

    # Update status
    info = self.dataset_navigator.get_dataset_info(dataset_id)
    if info:
        self.statusBar().showMessage(f"Switched to: {info.name}")

def _on_dataset_added(self, dataset_id: str, info_dict: dict):
    """Handle new dataset added."""
    self._update_dataset_menu()
    self.statusBar().showMessage(f"Dataset loaded: {info_dict.get('name', 'Unknown')}")

def _on_dataset_removed(self, dataset_id: str):
    """Handle dataset removed."""
    self._update_dataset_menu()

def _update_dataset_menu(self):
    """Update datasets submenu."""
    self.dataset_menu.clear()

    datasets = self.dataset_navigator.list_datasets()
    active_id = self.dataset_navigator.get_active_dataset_id()

    if not datasets:
        no_datasets = QAction("(No datasets loaded)", self)
        no_datasets.setEnabled(False)
        self.dataset_menu.addAction(no_datasets)
    else:
        for ds in datasets:
            action = QAction(ds['name'], self)
            action.setCheckable(True)
            action.setChecked(ds['dataset_id'] == active_id)
            action.setData(ds['dataset_id'])
            action.triggered.connect(
                lambda checked, did=ds['dataset_id']:
                self.dataset_navigator.set_active_dataset(did)
            )
            self.dataset_menu.addAction(action)

def _close_current_dataset(self):
    """Close the currently active dataset."""
    active_id = self.dataset_navigator.get_active_dataset_id()
    if active_id:
        self.dataset_navigator.remove_dataset(active_id)
        self.app_settings.remove_loaded_dataset(active_id)

def _restore_last_session(self):
    """Restore last session on startup."""
    # Restore datasets from settings
    saved_datasets = self.app_settings.get_loaded_datasets()
    restored = self.dataset_navigator.restore_from_serialized({
        'datasets': saved_datasets,
        'active_dataset_id': self.app_settings.get_active_dataset_id()
    })

    if restored > 0:
        # Restore session state
        session = self.app_settings.get_session_state()

        # Restore viewport
        vp = session.get('viewport', {})
        self.viewport_state.set_limits(
            vp.get('time_min', 0),
            vp.get('time_max', 1000),
            vp.get('trace_min', 0),
            vp.get('trace_max', 100)
        )

        # Restore display settings
        if 'colormap' in session:
            self.viewport_state.set_colormap(session['colormap'])
        if 'interpolation' in session:
            self.viewport_state.set_interpolation(session['interpolation'])

        # Restore navigation
        gather_id = session.get('current_gather_id', 0)
        if gather_id > 0:
            self.gather_navigator.goto_gather(gather_id)

        sort_keys = session.get('sort_keys', [])
        if sort_keys:
            self.gather_navigator.set_sort_keys(sort_keys)

        self.statusBar().showMessage(f"Restored {restored} dataset(s) from last session")
```

---

## Integration Checklist

### Phase 1: Core Integration (Required)

- [ ] Add `DatasetNavigator` instance to `MainWindow.__init__`
- [ ] Replace `self.settings = QSettings(...)` with `self.app_settings = get_settings()`
- [ ] Add `closeEvent` for session saving
- [ ] Add `_save_current_session` method
- [ ] Add `_restore_last_session` method
- [ ] Register datasets on import (`_on_segy_imported`)
- [ ] Register datasets on Zarr load (`_load_from_zarr`)

### Phase 2: UI Integration (Recommended)

- [ ] Add Datasets submenu to File menu
- [ ] Add dataset switcher dropdown to toolbar
- [ ] Add `_update_dataset_menu` method
- [ ] Add `_close_current_dataset` method
- [ ] Add keyboard shortcuts for dataset switching (Ctrl+Tab?)

### Phase 3: Signal Connections (Required)

- [ ] Connect `dataset_navigator.active_dataset_changed`
- [ ] Connect `dataset_navigator.dataset_added`
- [ ] Connect `dataset_navigator.dataset_removed`
- [ ] Connect `app_settings.session_restored`

---

## Files to Modify

| File | Changes Required |
|------|------------------|
| `main_window.py` | Major - add all integration code |
| `main.py` | Minor - no changes needed |
| `views/control_panel.py` | Optional - add dataset info display |
| `views/gather_navigation_panel.py` | Optional - show dataset name |

---

## Testing After Integration

1. **Start fresh app** - Should show empty state
2. **Load SEG-Y** - Should register in DatasetNavigator
3. **Load second SEG-Y** - Should have 2 datasets in menu
4. **Switch datasets** - Should update viewers
5. **Close app** - Should save session
6. **Reopen app** - Should restore last session
7. **Check datasets menu** - Should show previously loaded datasets

---

## Estimated Effort

| Task | Time |
|------|------|
| Core integration (Phase 1) | 2-3 hours |
| UI integration (Phase 2) | 1-2 hours |
| Testing & debugging | 1-2 hours |
| **Total** | **4-7 hours** |

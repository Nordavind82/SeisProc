#!/usr/bin/env python3
"""
Update dataset paths to match local SEG-Y file location.

This script:
1. Updates imported dataset metadata to point to local SEG-Y file
2. Syncs processed datasets to inherit metadata from parent dataset
3. Creates symlinks to match expected directory structure

Usage:
    python -m utils.update_dataset_paths [--dry-run]

Options:
    --dry-run   Show what would be changed without making changes
"""

import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime


# Configuration - adjust these paths as needed
SEISMIC_DATA_DIR = Path("/Users/olegadamovich/SeismicData")
LOCAL_SEGY_PATH = SEISMIC_DATA_DIR / "4proc_src_step3.sgy"
IMPORTED_DATASET = SEISMIC_DATA_DIR / "4proc_src_step3_20251208_111344"
PROCESSING_DIR = SEISMIC_DATA_DIR / "processing"


def backup_file(file_path: Path) -> Path:
    """Create timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f".backup_{timestamp}.json")
    shutil.copy2(file_path, backup_path)
    return backup_path


def update_imported_dataset(dry_run: bool = False) -> bool:
    """Update the imported dataset metadata to use local SEG-Y path."""
    metadata_path = IMPORTED_DATASET / "metadata.json"

    if not metadata_path.exists():
        print(f"ERROR: Metadata not found: {metadata_path}")
        return False

    if not LOCAL_SEGY_PATH.exists():
        print(f"ERROR: Local SEG-Y file not found: {LOCAL_SEGY_PATH}")
        return False

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get current paths
    old_source = metadata.get('seismic_metadata', {}).get('source_file', 'N/A')
    old_original = metadata.get('seismic_metadata', {}).get('original_segy_path', 'N/A')

    print(f"\n{'='*60}")
    print("IMPORTED DATASET UPDATE")
    print(f"{'='*60}")
    print(f"Dataset: {IMPORTED_DATASET.name}")
    print(f"Current source_file: {old_source}")
    print(f"Current original_segy_path: {old_original}")
    print(f"New path: {LOCAL_SEGY_PATH}")

    if dry_run:
        print("\n[DRY RUN] Would update paths")
        return True

    # Backup
    backup_path = backup_file(metadata_path)
    print(f"Backup created: {backup_path}")

    # Update paths
    local_path_str = str(LOCAL_SEGY_PATH.absolute())

    if 'seismic_metadata' not in metadata:
        metadata['seismic_metadata'] = {}

    metadata['seismic_metadata']['source_file'] = local_path_str
    metadata['seismic_metadata']['original_segy_path'] = local_path_str

    if 'file_info' in metadata['seismic_metadata']:
        metadata['seismic_metadata']['file_info']['filename'] = LOCAL_SEGY_PATH.name

    # Also add at top level for compatibility
    metadata['original_segy_path'] = local_path_str

    # Save
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✓ Metadata updated successfully")
    return True


def create_symlink(source: Path, target: Path, dry_run: bool = False) -> bool:
    """Create a symlink, handling existing files/links."""
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            if dry_run:
                print(f"    [DRY RUN] Would update symlink: {target.name}")
            else:
                target.unlink()
        else:
            print(f"    SKIP: {target.name} exists and is not a symlink")
            return False

    if dry_run:
        print(f"    [DRY RUN] Would create symlink: {target.name} -> {source}")
    else:
        os.symlink(source, target)
        print(f"    ✓ Symlink: {target.name}")
    return True


def sync_processed_datasets(dry_run: bool = False) -> int:
    """
    Sync processed datasets to inherit metadata from parent dataset.

    Processed datasets in /processing/ directory need:
    - metadata.json with parent reference
    - Proper original_segy_path for export
    - traces.zarr symlink at root (from output/)
    - headers.parquet symlink (from parent)
    - ensemble_index.parquet symlink (from parent)
    """
    if not PROCESSING_DIR.exists():
        print(f"\nNo processing directory found: {PROCESSING_DIR}")
        return 0

    # Load parent metadata
    parent_metadata_path = IMPORTED_DATASET / "metadata.json"
    if not parent_metadata_path.exists():
        print(f"ERROR: Parent metadata not found: {parent_metadata_path}")
        return 0

    with open(parent_metadata_path, 'r') as f:
        parent_metadata = json.load(f)

    updated_count = 0
    processed_dirs = [d for d in PROCESSING_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"\n{'='*60}")
    print("PROCESSED DATASETS SYNC")
    print(f"{'='*60}")
    print(f"Found {len(processed_dirs)} processed dataset(s)")

    for proc_dir in sorted(processed_dirs):
        print(f"\n--- {proc_dir.name} ---")

        metadata_path = proc_dir / "metadata.json"
        output_dir = proc_dir / "output"
        traces_zarr_source = output_dir / "traces.zarr" if output_dir.exists() else None

        # Check if has traces in output/
        if not traces_zarr_source or not traces_zarr_source.exists():
            # Also check if already at root level
            if (proc_dir / "traces.zarr").exists():
                print(f"  traces.zarr already at root level")
                traces_zarr_source = proc_dir / "traces.zarr"
            else:
                print(f"  SKIP: No traces.zarr found")
                continue

        # Create symlinks for directory structure
        print(f"  Creating symlinks:")

        # 1. traces.zarr symlink (from output/ to root)
        traces_target = proc_dir / "traces.zarr"
        if traces_zarr_source != traces_target:
            create_symlink(traces_zarr_source, traces_target, dry_run)

        # 2. headers.parquet symlink (from parent)
        parent_headers = IMPORTED_DATASET / "headers.parquet"
        if parent_headers.exists():
            create_symlink(parent_headers, proc_dir / "headers.parquet", dry_run)

        # 3. ensemble_index.parquet symlink (from parent)
        parent_ensemble = IMPORTED_DATASET / "ensemble_index.parquet"
        if parent_ensemble.exists():
            create_symlink(parent_ensemble, proc_dir / "ensemble_index.parquet", dry_run)

        # 4. trace_index.parquet symlink (from parent)
        parent_trace_idx = IMPORTED_DATASET / "trace_index.parquet"
        if parent_trace_idx.exists():
            create_symlink(parent_trace_idx, proc_dir / "trace_index.parquet", dry_run)

        # Load or create metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  Existing metadata found")
        else:
            # Create new metadata based on parent
            metadata = {
                "parent_dataset": str(IMPORTED_DATASET.absolute()),
                "parent_dataset_name": IMPORTED_DATASET.name,
                "processing_session": proc_dir.name,
            }
            print(f"  Creating new metadata")

        # Inherit key fields from parent
        for key in ['shape', 'sample_rate', 'n_samples', 'n_traces', 'duration_ms', 'nyquist_freq']:
            if key in parent_metadata:
                metadata[key] = parent_metadata[key]

        # Set SEG-Y path reference
        local_path_str = str(LOCAL_SEGY_PATH.absolute())
        metadata['original_segy_path'] = local_path_str

        if 'seismic_metadata' not in metadata:
            metadata['seismic_metadata'] = {}
        metadata['seismic_metadata']['original_segy_path'] = local_path_str
        metadata['seismic_metadata']['source_file'] = local_path_str

        # Copy header mapping from parent if available
        if 'seismic_metadata' in parent_metadata and 'header_mapping' in parent_metadata['seismic_metadata']:
            metadata['seismic_metadata']['header_mapping'] = parent_metadata['seismic_metadata']['header_mapping']

        if dry_run:
            print(f"  [DRY RUN] Would write metadata.json")
        else:
            # Backup if exists
            if metadata_path.exists():
                backup_file(metadata_path)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Metadata written")

        updated_count += 1

    return updated_count


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No changes will be made")
        print("="*60)

    print(f"\nConfiguration:")
    print(f"  Seismic data dir: {SEISMIC_DATA_DIR}")
    print(f"  Local SEG-Y file: {LOCAL_SEGY_PATH}")
    print(f"  Imported dataset: {IMPORTED_DATASET}")
    print(f"  Processing dir:   {PROCESSING_DIR}")

    # Check prerequisites
    if not LOCAL_SEGY_PATH.exists():
        print(f"\nERROR: Local SEG-Y file not found: {LOCAL_SEGY_PATH}")
        sys.exit(1)

    # Update imported dataset
    success = update_imported_dataset(dry_run)
    if not success:
        print("\nFailed to update imported dataset")
        sys.exit(1)

    # Sync processed datasets
    count = sync_processed_datasets(dry_run)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Imported dataset: Updated")
    print(f"Processed datasets synced: {count}")

    if dry_run:
        print(f"\nRun without --dry-run to apply changes")
    else:
        print(f"\n✓ All updates complete")


if __name__ == '__main__':
    main()

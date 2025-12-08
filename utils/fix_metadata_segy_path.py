#!/usr/bin/env python3
"""
Utility to add original_segy_path to processed dataset metadata.

When a dataset was processed before the fix that preserves the original
SEG-Y path, this utility can retroactively add the path to enable export.

Usage:
    python -m utils.fix_metadata_segy_path /path/to/processed/dataset /path/to/original.sgy

    Or interactively:
    python -m utils.fix_metadata_segy_path /path/to/processed/dataset
"""

import json
import sys
from pathlib import Path


def fix_metadata_segy_path(dataset_dir: str, original_segy_path: str) -> bool:
    """
    Add original_segy_path to a processed dataset's metadata.

    Args:
        dataset_dir: Path to the processed dataset directory (contains metadata.json)
        original_segy_path: Path to the original SEG-Y file

    Returns:
        True if successful, False otherwise
    """
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / 'metadata.json'

    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {dataset_dir}")
        return False

    segy_path = Path(original_segy_path)
    if not segy_path.exists():
        print(f"WARNING: Original SEG-Y file not found at {original_segy_path}")
        print("The path will be saved anyway, but export may fail if the file is not accessible.")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return False

    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check if already has the path
    existing_path = metadata.get('original_segy_path')
    if existing_path:
        print(f"Metadata already has original_segy_path: {existing_path}")
        response = input(f"Replace with {original_segy_path}? [y/N]: ")
        if response.lower() != 'y':
            print("Keeping existing path.")
            return True

    # Add the path at top level
    metadata['original_segy_path'] = str(segy_path.absolute())

    # Also add to seismic_metadata for consistency
    if 'seismic_metadata' not in metadata:
        metadata['seismic_metadata'] = {}
    metadata['seismic_metadata']['original_segy_path'] = str(segy_path.absolute())

    # Backup original metadata
    backup_path = metadata_path.with_suffix('.json.backup')
    import shutil
    shutil.copy2(metadata_path, backup_path)
    print(f"Backed up original metadata to: {backup_path}")

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"SUCCESS: Added original_segy_path to metadata")
    print(f"  Dataset: {dataset_dir}")
    print(f"  SEG-Y Path: {segy_path.absolute()}")
    print(f"\nYou should now be able to export this dataset to SEG-Y.")

    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dataset_dir = sys.argv[1]

    if len(sys.argv) >= 3:
        original_segy_path = sys.argv[2]
    else:
        # Interactive mode
        print(f"Dataset directory: {dataset_dir}")
        original_segy_path = input("Enter path to original SEG-Y file: ").strip()

        if not original_segy_path:
            print("ERROR: No path provided")
            sys.exit(1)

    success = fix_metadata_segy_path(dataset_dir, original_segy_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

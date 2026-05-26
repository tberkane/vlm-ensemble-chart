#!/usr/bin/env python3
"""
Create a random 10% subset of ChartQA test set.
Copies PNG images and corresponding table CSV files to data/ChartQA Dataset/test_10_percent/
"""

import random
import shutil
from pathlib import Path


def create_chartqa_subset():
    """Create a random 10% subset of ChartQA test set."""

    # Define paths
    base_dir = Path("data/World Bank")
    test_png_dir = base_dir / "test" / "png"
    test_tables_dir = base_dir / "test" / "tables"
    output_dir = base_dir / "test_50_percent"
    output_png_dir = output_dir / "png"
    output_tables_dir = output_dir / "tables"

    # Check if source directories exist
    if not test_png_dir.exists():
        print(f"Error: Source directory not found: {test_png_dir}")
        return False

    if not test_tables_dir.exists():
        print(f"Error: Source directory not found: {test_tables_dir}")
        return False

    # Get all PNG files
    png_files = list(test_png_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files in test set")

    if len(png_files) == 0:
        print("Error: No PNG files found in test set")
        return False

    # Calculate 10% subset size
    subset_size = max(1, int(len(png_files) * 0.5))
    print(f"Selecting {subset_size} files (50% of {len(png_files)})")

    # Randomly sample files
    random.seed(42)  # For reproducibility
    selected_pngs = random.sample(png_files, subset_size)

    # Create output directories
    output_png_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    copied_count = 0
    missing_table_count = 0

    for png_file in selected_pngs:
        # Get the base filename (without extension)
        base_name = png_file.stem

        # Copy PNG file
        dest_png = output_png_dir / png_file.name
        shutil.copy2(png_file, dest_png)

        # Find and copy corresponding table file
        table_file = test_tables_dir / f"{base_name}.csv"
        if table_file.exists():
            dest_table = output_tables_dir / table_file.name
            shutil.copy2(table_file, dest_table)
            copied_count += 1
        else:
            print(f"Warning: Table file not found for {base_name}.png")
            missing_table_count += 1

    print(f"\n✓ Successfully created subset:")
    print(f"  - Copied {copied_count} image-table pairs")
    if missing_table_count > 0:
        print(f"  - Warning: {missing_table_count} images without corresponding tables")
    print(f"  - Output directory: {output_dir}")
    print(f"    - Images: {output_png_dir}")
    print(f"    - Tables: {output_tables_dir}")

    return True


if __name__ == "__main__":
    create_chartqa_subset()

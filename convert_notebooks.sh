#!/bin/bash

# Script to convert all Jupyter notebooks to Python scripts
# Usage: ./convert_notebooks.sh

# Create the converted subdirectory if it doesn't exist
mkdir -p converted

# Check if jupyter/nbconvert is available
if ! command -v jupyter &> /dev/null; then
    echo "Error: jupyter is not installed or not in PATH"
    echo "Please install jupyter: pip install jupyter"
    exit 1
fi

# Find all .ipynb files in current directory and convert them
converted_count=0
failed_count=0

echo "Converting Jupyter notebooks to Python scripts..."
echo "=========================================="

for notebook in *.ipynb; do
    # Check if any .ipynb files exist
    if [ ! -e "$notebook" ]; then
        echo "No Jupyter notebook files (*.ipynb) found in current directory"
        exit 0
    fi

    # Get filename without extension
    basename=$(basename "$notebook" .ipynb)
    output_file="converted/${basename}.py"

    echo "Converting: $notebook -> $output_file"

    # Convert notebook to python script
    if jupyter nbconvert --to python --output-dir=converted "$notebook" 2>/dev/null; then
        ((converted_count++))
        echo "✓ Successfully converted $notebook"
    else
        ((failed_count++))
        echo "✗ Failed to convert $notebook"
    fi
done

echo "=========================================="
echo "Conversion complete!"
echo "Successfully converted: $converted_count files"
if [ $failed_count -gt 0 ]; then
    echo "Failed conversions: $failed_count files"
fi
echo "Output directory: ./converted/"

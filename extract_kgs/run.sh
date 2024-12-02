#!/bin/bash

# Set the base directory
BASE_DIR="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

# Run index.py
echo "Running index.py..."
python3 "$BASE_DIR/extract_kgs/exp4_1qa/index.py"

# Run aggregate.py
echo "Running aggregate.py..."
python3 "$BASE_DIR/extract_kgs/exp4_1qa/aggregate.py"

# Run cluster.py
echo "Running cluster.py..."
python3 "$BASE_DIR/extract_kgs/exp4_1qa/cluster.py"

echo "All scripts completed."

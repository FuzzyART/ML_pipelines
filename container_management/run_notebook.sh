#!/bin/bash
set -e

CONTAINER_NAME="$1"
INPUT_NB="$2"
OUTPUT_NB="${3:-out.ipynb}"
SKIP_TAG="${4:-}"

if [ -z "$CONTAINER_NAME" ] || [ -z "$INPUT_NB" ]; then
    echo "Usage: $0 <container_name> <input_notebook> [output_notebook] [skip_tag]"
    echo "Example: $0 pipeline_nlp my_notebook.ipynb results/out.ipynb skip_cell"
    exit 1
fi

# Ensure container is actively running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' is not running. Please execute setup.sh first."
    exit 1
fi

# Create target output directory inside /workDir if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_NB")
docker exec -i "$CONTAINER_NAME" mkdir -p "/workDir/$OUTPUT_DIR"

echo "==> Running Papermill on [$CONTAINER_NAME] ($INPUT_NB -> $OUTPUT_NB)..."

if [ -n "$SKIP_TAG" ]; then
    echo "--> Skipping cells tagged with: '$SKIP_TAG'"
    docker exec -i "$CONTAINER_NAME" bash -c "
        source /opt/venv/bin/activate && \
        papermill '/workDir/$INPUT_NB' '/workDir/$OUTPUT_NB' -k opt_venv --skip-tagged-notebook-cells '$SKIP_TAG'
    "
else
    docker exec -i "$CONTAINER_NAME" bash -c "
        source /opt/venv/bin/activate && \
        papermill '/workDir/$INPUT_NB' '/workDir/$OUTPUT_NB' -k opt_venv
    "
fi

echo "==> Notebook execution completed successfully on [$CONTAINER_NAME]! Saved to /workDir/$OUTPUT_NB"

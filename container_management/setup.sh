#!/bin/bash
set -e

CONTAINER_NAME="$1"
PROJECT_PATH="$2"

if [ -z "$CONTAINER_NAME" ] || [ -z "$PROJECT_PATH" ]; then
    echo "Usage: $0 <container_name> <path_to_project> [extra_package_dir1] [extra_package_dir2] ..."
    echo "Example: $0 pipeline_nlp ../nlp_project ../shared_helpers ../maturin_pkg"
    exit 1
fi

# Resolve absolute path for project directory
PROJECT_ABS="$(cd "$PROJECT_PATH" && pwd)"

# Build array for extra mounts and PYTHONPATH environment variables
EXTRA_MOUNTS=""
EXTRA_PYTHONPATHS=""
SHIFT_ARGS=("${@:3}")

for idx in "${!SHIFT_ARGS[@]}"; do
    pkg_path="${SHIFT_ARGS[$idx]}"
    if [ -d "$pkg_path" ]; then
        pkg_abs="$(cd "$pkg_path" && pwd)"
        target_dir="/extra_pkgs/pkg_$idx"
        EXTRA_MOUNTS="$EXTRA_MOUNTS -v ${pkg_abs}:${target_dir}"
        
        if [ -z "$EXTRA_PYTHONPATHS" ]; then
            EXTRA_PYTHONPATHS="${target_dir}"
        else
            EXTRA_PYTHONPATHS="${EXTRA_PYTHONPATHS}:${target_dir}"
        fi
        echo "--> Staging extra package directory: $pkg_abs -> $target_dir"
    fi
done

echo "==> Starting ROCm Container [$CONTAINER_NAME]..."

# Stop and replace container if it already exists with this name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -dt --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  -v /dev/shm:/hostShm \
  -v "${PROJECT_ABS}:/workDir" \
  $EXTRA_MOUNTS \
  -w /workDir \
  --name "$CONTAINER_NAME" \
  rocm/pytorch \
  /bin/bash

echo "==> Initializing persistent /opt/venv inside [$CONTAINER_NAME]..."
docker exec -i "$CONTAINER_NAME" bash << EOF
set -e

# Locate Python binary containing ROCm PyTorch
PY_BIN=""
for py in \$(which -a python python3 python3.10 python3.11 2>/dev/null); do
    if \$py -c "import torch" 2>/dev/null; then
        PY_BIN="\$py"
        break
    fi
done

if [ -z "\$PY_BIN" ]; then
    echo "Error: Could not find Python interpreter with PyTorch installed."
    exit 1
fi

echo "Found ROCm PyTorch under: \$PY_BIN"

# Create /opt/venv linked to base system site-packages
\$PY_BIN -m venv --system-site-packages /opt/venv
source /opt/venv/bin/activate

# Upgrade build tools for maturin/custom packages
pip install --quiet --upgrade pip setuptools wheel maturin papermill ipykernel

# Install requirements from project directory if available
if [ -f "/workDir/requirements.txt" ]; then
    echo "--> Installing /workDir/requirements.txt..."
    pip install --quiet -r /workDir/requirements.txt
fi

# Register /opt/venv as Jupyter Kernel
python -m ipykernel install --sys-prefix --name opt_venv --display-name "Python (opt_venv)"

# Persist PYTHONPATH for mounted extra packages across subshells
if [ -n "$EXTRA_PYTHONPATHS" ]; then
    echo "export PYTHONPATH=\"\$PYTHONPATH:$EXTRA_PYTHONPATHS\"" >> /opt/venv/bin/activate
fi

echo "==> [$CONTAINER_NAME] setup complete!"
EOF

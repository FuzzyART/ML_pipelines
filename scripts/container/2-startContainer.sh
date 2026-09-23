#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

workDirFolder=${SCRIPT_DIR}/../../

docker run -dt --rm  \
--network=host \
--device=/dev/kfd \
--device=/dev/dri \
--security-opt seccomp=unconfined \
--ipc=host \
--shm-size 16G \
--group-add video \
--cap-add=SYS_PTRACE \
-v /dev/shm:/hostShm \
-v ${workDirFolder}:/workDir/ \
--name ml_pipeline_cont \
ml_pipeline_cont \
fish

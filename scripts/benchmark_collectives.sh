#!/bin/bash

# Benchmark a communication pattern for various world sizes and dtypes
# Generates multiple CSV/PNG files

BASE_DIR=$(dirname "$0")

# Install plotting deps (see requirements_plotting.txt)
pip install -r "${BASE_DIR}/../requirements_plotting.txt"


export DEVICE="npu"
#DTYPES="int float16 float32"
DTYPES="float16"

# To surpress a torchrun warning
export OMP_NUM_THREADS=1


COLLECTIVES="broadcast allreduce allgather reducescatter alltoall reduce scatter"

for BENCH in ${COLLECTIVES}
do
    for DTYPE in ${DTYPES}
    do
        export BENCH=${BENCH}
        export HCCL_DTYPE=${DTYPE}
        for WORLD_SIZE in {2..8}
        do
            echo "${BENCH} (world_size: ${WORLD_SIZE} | dtype: ${DTYPE})"
            torchrun --nnodes 1 --nproc_per_node "${WORLD_SIZE}" \
                pytorch_hccl_tests/cli.py --benchmark "${BENCH}" \
                                          --device ${DEVICE} \
                                          --dtype "${HCCL_DTYPE}"
        done

        python "${BASE_DIR}/plotter_collectives.py"
    done
done

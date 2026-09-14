import logging

import pandas as pd
import torch.distributed as dist

from pytorch_hccl_tests.commons import (
    elaspsed_time_ms,
    get_device,
    get_device_event,
    get_nbytes_from_dtype,
    safe_rand,
    sync_device,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def latency(args):
    backend = args.backend
    rank = dist.get_rank()
    dtype = args.dtype
    world_size = dist.get_world_size()
    device = get_device(backend, rank)
    pg = None

    options = Options("Latency", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    # Print header
    if rank == 0:
        logger.info(f"# PyTorch Benchmark {options.benchmark} Test")
        logger.info(f'# {"Size (B)":<8}{"Latency (ms)":>18}{"BW (GB/s)":>18}')

    rows = []

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large
        iterations = range(options.iterations + options.skip)
        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        start_event = end_event = None
        dist.barrier()
        if rank == 0:
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.send(s_msg, 1, pg, 1)
                dist.recv(r_msg, 1, pg, 1)
            end_event = get_device_event(backend)
            sync_device(backend)
        elif rank == 1:
            for i in iterations:
                if i == options.skip:
                    start_event = get_device_event(backend)
                dist.recv(r_msg, 0, pg, 1)
                dist.send(s_msg, 0, pg, 1)
            end_event = get_device_event(backend)
            sync_device(backend)
        dist.barrier()

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)

        # Divide by 2 since one messsage sent and one message received
        avg_latency_ms = (
            Utils.avg_lat(total_time_ms, options.iterations, world_size, device) / 2
        )

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            # One-way bandwidth — avg_latency_ms is already one-way latency
            # (the original code divides round-trip by 2 on lines 70-72).
            t_oneway_sec = avg_latency_ms / 1000.0
            bw_gbps = (size_in_bytes / 1e9) / t_oneway_sec if t_oneway_sec > 0 else 0.0
            logger.info(f"{size_in_bytes:<10d}{avg_latency_ms:>18.2f}{bw_gbps:>18.2f}")
            new_row = {
                "size_in_bytes": size_in_bytes,
                "avg_latency_ms": avg_latency_ms,
                "bw_gbps": bw_gbps,
            }
            rows.append(new_row)

    # Persist result to CSV file
    if rank == 0:
        pd.DataFrame(rows).to_csv(
            f"osu_latency-{device.type}-{dtype}-{world_size}.csv", index=False
        )

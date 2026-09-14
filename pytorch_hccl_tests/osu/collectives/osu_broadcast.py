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


def broadcast(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, args.local_rank)
    pg = None

    options = Options("Broadcast", args)
    Utils.check_numprocs(world_size, rank, limit=3)
    Utils.print_header(options.benchmark, rank)

    rows = []

    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        # safe_rand is a wrapper of torch.rand for floats and
        # torch.randint for integral types
        msg = safe_rand(size, dtype=dtype).to(device)

        start_event = None
        dist.barrier()
        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            dist.broadcast(msg, 0, pg, False)
        end_event = get_device_event(backend)
        sync_device(backend)
        dist.barrier()

        total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
        # Broadcast is one-shot (root → all), not a round-trip.
        avg_latency_ms = Utils.avg_lat(
            total_time_ms, options.iterations, world_size, device
        )

        if rank == 0:
            logger.info(f"{size:<10d}{avg_latency_ms:>18.2f}")
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            new_row = {"size_in_bytes": size_in_bytes, "avg_latency_ms": avg_latency_ms}
            rows.append(new_row)

    if rank == 0:
        pd.DataFrame(rows).to_csv(
            f"osu_broadcast-{device.type}-{dtype}-{world_size}.csv", index=False
        )

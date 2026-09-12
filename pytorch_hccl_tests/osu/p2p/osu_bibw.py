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
    wait_all,
)
from pytorch_hccl_tests.osu.options import Options
from pytorch_hccl_tests.osu.osu_util_mpi import Utils

logger = logging.getLogger(__name__)


def _exchange_window(
    s_msg, r_msg, partner, send_tag, recv_tag, window_sizes, send_first, pg=None
):
    """Post a window of non-blocking sends/recvs and wait for all of them.

    ``send_first`` controls the posting order: HCCL matches send/recv by
    posting position, so the two ranks must post in mirrored order.
    """
    send_requests = [None] * len(window_sizes)
    recv_requests = [None] * len(window_sizes)

    def post_sends():
        for j in window_sizes:
            send_requests[j] = dist.isend(s_msg, partner, pg, send_tag)

    def post_recvs():
        for j in window_sizes:
            recv_requests[j] = dist.irecv(r_msg, partner, pg, recv_tag)

    if send_first:
        post_sends()
        post_recvs()
    else:
        post_recvs()
        post_sends()

    wait_all(send_requests)
    wait_all(recv_requests)


def bibw(args):
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    options = Options("Bi-Directional Bandwidth", args)
    Utils.check_numprocs(world_size, rank, limit=2)

    if rank == 0:
        logger.info(f"# OMB-Py MPI {options.benchmark} Test")
        logger.info(f'# {"Size (B)":<8}{"Bandwidth (GB/s)":>18}')

    rows = []

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        window_sizes = list(range(window_size))

        # Tags are swapped between ranks (canonical OSU C / mpi4py).
        # Rank 0 posts sends first, rank 1 posts recvs first — HCCL matches
        # send/recv by posting position, so the first op on each rank must
        # form a valid send↔recv pair.
        # https://github.com/mpi4py/mpi4py/blob/1c1d41/demo/osu_bibw.py
        # https://mvapich.cse.ohio-state.edu/benchmarks/
        if rank == 0:
            partner = 1
            recv_tag = 10
            send_tag = 100
        else:
            partner = 0
            recv_tag = 100
            send_tag = 10

        s_msg = safe_rand(size, dtype=dtype).to(device)
        r_msg = safe_rand(size, dtype=dtype).to(device)

        dist.barrier()
        start_event = None
        for i in range(options.iterations + options.skip):
            if i == options.skip:
                start_event = get_device_event(backend)
            _exchange_window(
                s_msg,
                r_msg,
                partner,
                send_tag,
                recv_tag,
                window_sizes,
                send_first=rank == 0,
                pg=pg,
            )
        end_event = get_device_event(backend)
        sync_device(backend)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)

            # Canonical OSU bidirectional bandwidth formula.
            # Reports aggregate bandwidth across both directions.
            # The × 2 accounts for both send and recv traffic at each
            # iteration (osu_bibw.c line 113: ((size / 1.0e6) * loop
            # * window_size * 2)).
            total_time_ms = elaspsed_time_ms(backend, start_event, end_event)
            t_sec = total_time_ms / 1000.0
            bw_gbps = (
                size_in_bytes * options.iterations * window_size * 2 / (1e9 * t_sec)
            )

            logger.info(f"{size_in_bytes:<10d}{bw_gbps:>18.2f}")
            new_row = {
                "size_in_bytes": int(size_in_bytes),
                "bw_gbps": bw_gbps,
            }
            rows.append(new_row)

    if rank == 0:
        pd.DataFrame(rows).to_csv(
            f"osu_bibw_gbps-{device.type}-{dtype}-{world_size}.csv", index=False
        )

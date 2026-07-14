import logging

import pandas as pd
import torch
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


def mbw_mr(args):
    """OSU-style multiple bandwidth / message rate benchmark (osu_mbw_mr).

    Measures the *aggregate* uni-directional bandwidth and message rate
    achieved when ``world_size // 2`` process pairs communicate
    concurrently. Ranks are block-assigned: rank ``r`` in the first half
    sends to its partner ``r + pairs`` in the second half (the same pairing
    as ``osu_multi_lat``). Each sender pushes a window of back-to-back
    non-blocking messages (as in ``osu_bw``) and waits for a single
    acknowledgement before the next window. An odd leftover rank, if any,
    only joins the collective barrier/reduce and performs no transfers.
    """
    backend = args.backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = args.dtype
    device = get_device(backend, rank)
    pg = None

    if world_size < 2:
        raise SystemExit(
            "mbw_mr requires at least two processes" if rank == 0 else None
        )

    options = Options("Multiple Bandwidth / Message Rate", args)

    pairs = world_size // 2
    is_sender = rank < pairs
    is_receiver = pairs <= rank < 2 * pairs
    partner = rank + pairs if is_sender else rank - pairs

    if rank == 0:
        logger.info("# OMB-Py MPI %s Test" % (options.benchmark))
        logger.info("# concurrent pairs: %d" % pairs)
        logger.info(
            "# %-8s%20s%20s" % ("Size (B)", "Aggregate BW (GB/s)", "Msg Rate (M/s)")
        )

    rows = []

    window_size = 64
    for size in Utils.message_sizes(options):
        if size > options.large_message_size:
            options.skip = options.skip_large
            options.iterations = options.iterations_large

        window_sizes = range(window_size)
        requests = [None] * window_size

        # Per-rank elapsed time; only senders time their loop. Receivers and
        # any leftover rank contribute 0 to the reduction below.
        local_t_sec = 0.0

        dist.barrier()
        if is_sender:
            # safe_rand is a wrapper of torch.rand for floats and
            # torch.randint for integral types.
            s_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            r_msg = safe_rand(4, dtype=dtype).to(device)
            for i in range(options.iterations + options.skip):
                if i == options.skip:
                    start_event = get_device_event(backend)
                for j in window_sizes:
                    requests[j] = dist.isend(s_msg[j], partner, pg, 100)
                wait_all(requests)
                dist.recv(r_msg, partner, pg, 101)
            end_event = get_device_event(backend)
            sync_device(backend)
            local_t_sec = elaspsed_time_ms(backend, start_event, end_event) / 1000.0
        elif is_receiver:
            s_msg = safe_rand(4, dtype=dtype).to(device)
            r_msg = [
                safe_rand(size, dtype=dtype).to(device) for _ in range(window_size)
            ]
            for i in range(options.iterations + options.skip):
                for j in window_sizes:
                    requests[j] = dist.irecv(r_msg[j], partner, pg, 100)
                wait_all(requests)
                dist.send(s_msg, partner, pg, 101)
        # leftover rank (odd world_size) only joins the collectives below

        # All ranks participate in the reduction: senders contribute their
        # measured time; non-senders contribute 0.
        t_sum = torch.tensor(local_t_sec, dtype=torch.float32).to(device)
        dist.reduce(t_sum, 0, op=dist.ReduceOp.SUM)

        if rank == 0:
            size_in_bytes = int(size) * get_nbytes_from_dtype(dtype)
            messages = options.iterations * window_size

            # Canonical OSU osu_mbw_mr aggregation: divide the total bytes /
            # messages moved by ALL pairs by the average per-pair time.
            #   avg_t       = sum(sender_time) / pairs
            #   agg_bw_gbps = (pairs * size_bytes * messages) / (1e9 * avg_t)
            #   msg_rate    = (pairs * messages) / avg_t
            # See osu_mbw_mr.c and https://mvapich.cse.ohio-state.edu/benchmarks/
            avg_t = t_sum.item() / pairs
            agg_bw_gbps = (pairs * size_in_bytes * messages) / (1e9 * avg_t)
            msg_rate_mmps = (pairs * messages) / avg_t / 1e6

            logger.info(
                "%-10d%20.2f%20.2f" % (size_in_bytes, agg_bw_gbps, msg_rate_mmps)
            )
            rows.append(
                {
                    "size_in_bytes": int(size_in_bytes),
                    "agg_bw_gbps": agg_bw_gbps,
                    "msg_rate_mmps": msg_rate_mmps,
                    "pairs": pairs,
                }
            )

    # Persist result to CSV file
    if rank == 0:
        pd.DataFrame(rows).to_csv(
            f"osu_mbw_mr_gbps-{device.type}-{dtype}-{world_size}.csv", index=False
        )

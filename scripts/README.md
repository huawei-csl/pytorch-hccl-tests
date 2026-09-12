# scripts

Helper scripts for running benchmark sweeps, plotting their results, and launching the
benchmarks on ModelArts.

All scripts are meant to be run from the repository root, since the plotters read the CSV
files from the current working directory and write the PNG files next to them.

## Requirements

```bash
pip install -r requirements_plotting.txt
```

This installs `matplotlib` and `seaborn`; `pandas` and `torch` already come with
`pip install .` / `make install-npu-*`.

## Sweep drivers

| Script | What it does |
| --- | --- |
| `benchmark_latency_bandwidth.sh` | Runs `latency` and `bandwidth` on 2 ranks for each dtype in `DTYPES`, then calls the matching plotter. |
| `benchmark_collectives.sh` | Runs `broadcast allreduce allgather reducescatter alltoall` for world sizes 2..8 and each dtype, then calls `plotter_collectives.py`. |

Both default to `DEVICE=npu` and `DTYPES="float16"`; edit the variables at the top of the
script to sweep more dtypes (`int float16 float32`).

```bash
./scripts/benchmark_latency_bandwidth.sh
./scripts/benchmark_collectives.sh
```

## Plotters

Each plotter reads the CSV files produced by the benchmarks
(`osu_<benchmark>-<device>-<dtype>-<world_size>.csv`) and writes a PNG into the current
directory. They are configured through environment variables:

| Variable | Default | Used by |
| --- | --- | --- |
| `DEVICE` | `cpu` | all plotters |
| `HCCL_DTYPE` | `float16` | all plotters |
| `BENCH` | `allreduce` | `plotter_collectives.py` (which collective to plot) |

| Script | Input CSV | Output PNG |
| --- | --- | --- |
| `plotter_latency.py` | `osu_latency-<device>-<dtype>-2.csv` | `plot-latency-<device>-<dtype>.png` |
| `plotter_bandwidth.py` | `osu_bandwidth_gbps-<device>-<dtype>-2.csv` | `plot-bandwidth-<device>-<dtype>.png` |
| `plotter_collectives.py` | `osu_<BENCH>-<device>-<dtype>-<2..8>.csv` | `plot-<BENCH>-<device>-<dtype>.png` |
| `plotter_mbw_mr.py` | `osu_mbw_mr_gbps-<device>-<dtype>-<2..8>.csv` | `plot-mbw_mr-bw-...png`, `plot-mbw_mr-msgrate-...png` |

`plotter_collectives.py` and `plotter_mbw_mr.py` plot one series per world size and simply
skip the world sizes whose CSV is missing.

Example: sweep the multi-pair bandwidth / message rate benchmark and plot it.

```bash
export DEVICE=npu HCCL_DTYPE=float16
for ws in 2 4 6 8; do
    WORLD_SIZE=$ws make mbw-mr DEVICE=$DEVICE
done
python scripts/plotter_mbw_mr.py
```

## ModelArts

`ma_launch_pt_benchs.py` is the entrypoint used for multi-node runs on ModelArts. It derives the
rendezvous endpoint and node rank from the ModelArts job environment (`MA_VJ_NAME`, `MA_TASK_NAME`,
`MA_JOB_DIR`, `VC_TASK_INDEX`/`MA_TASK_INDEX`) and then launches `pytorch_hccl_tests/cli.py` with
`torchrun --device npu`.

| Argument | Env fallback | Default |
| --- | --- | --- |
| `--benchmark` | `PT_HCCL_BENCHMARK` | `allreduce` |
| `--code-dir` | `PT_HCCL_CODE_DIR` | `pytorch-hccl-benchs` |
| `--nnodes` | `MA_NUM_HOSTS` | `1` |
| `--nproc_per_node` | — | `8` |


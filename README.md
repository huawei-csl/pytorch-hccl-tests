# pytorch-hccl-tests

HCCL benchmarks based on the PyTorch/Ascend adapter (`torch-npu`). The benchmarks are ports of the ones proposed in the article [OMB-Py: Python Micro-Benchmarks for Evaluating Performance of MPI Libraries on HPC Systems](https://arxiv.org/pdf/2110.10659.pdf).

All benchmarks run on top of `torch.distributed`, so they work on `npu` (HCCL backend), `cuda` (NCCL) and `cpu` (Gloo). The point-to-point benchmarks (`latency`, `bandwidth`, `bibw`, `multi-latency`, `mbw_mr`) and the collectives (`allreduce`, `allgather`, `alltoall`, `broadcast`, `barrier`, `reducescatter`, `gather`, `reduce`, `scatter`) are supported.

Requires Python >= 3.9 and PyTorch 2.9.

### Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_dev.txt
make install
```

`make install` pulls the CPU build of PyTorch. To benchmark real accelerators use one of

```bash
make install-npu-arm   # Ascend/NPU, aarch64  (torch-npu 2.9.0)
make install-npu-x86   # Ascend/NPU, x86_64   (torch-npu 2.9.0)
make install-cuda      # CUDA
```

### Benchmark suites

The following benchmarks are available. To view the list of available benchmarks, type `make`

```
> make
...
<development related Make commands>
...
hello                OSU MPI/HCCL hello init benchmark
latency              OSU MPI/HCCL latency benchmark
multi-latency        OSU MPI/HCCL multi-latency benchmark
bandwidth            OSU MPI/HCCL bandwidth benchmark
bidirectional-bw     OSU MPI/HCCL bidirectional bandwidth benchmark
mbw-mr               OSU MPI/HCCL multiple bandwidth / message rate benchmark (multi-pair)
allreduce            OSU MPI/HCCL allreduce benchmark
allgather            OSU MPI/HCCL allgather benchmark
alltoall             OSU MPI/HCCL alltoall benchmark
barrier              OSU MPI/HCCL barrier benchmark
broadcast            OSU MPI/HCCL broadcast benchmark
gather               OSU MPI/HCCL Bandwidth benchmark
reduce               OSU MPI/HCCL Bandwidth benchmark
scatter              OSU MPI/HCCL Bandwidth benchmark
reducescatter        OSU MPI/HCCL reduce_scatter benchmark
collectives          OSU MPI/HCCL collective communications benchmark suite
benchmarks           OSU MPI/HCCL complete benchmark suite
```

In addition, `make p2p` runs the point-to-point suite (`latency bandwidth bidirectional-bw multi-latency mbw-mr`), and `make benchmarks` runs `p2p` plus `collectives`.

#### Configuration

The Make targets honour the following environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cpu` | `cpu`, `npu` or `cuda` |
| `WORLD_SIZE` | `2` | Number of ranks for the benchmarks that scale (collectives, `multi-latency`, `mbw-mr`) |
| `HCCL_DTYPE` | `float16` | Tensor dtype used by the benchmark |

```bash
make allreduce -e DEVICE=npu
WORLD_SIZE=8 make mbw-mr DEVICE=npu
```

Benchmarks can also be launched directly through `torchrun`, which additionally exposes
`--min`, `--max`, `--skip` (warmup iterations) and `--iterations`:

```bash
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/cli.py \
    --benchmark latency --device npu --dtype float16 --max 1048576
```

#### Example benchmark: Latency

```bash
make latency -e DEVICE=npu (default: cpu)
```

should output

```bash
> export OMP_NUM_THREADS=1
> make latency
torchrun --nnodes 1 --nproc_per_node 2 pytorch_hccl_tests/cli.py --benchmark latency --device cpu
[2026-05-01 10:27:19,729] {cli.py:68} INFO - ******************************
[2026-05-01 10:27:19,729] {cli.py:69} INFO - Selected benchmark : latency
[2026-05-01 10:27:19,729] {cli.py:70} INFO - Input device param : cpu
[2026-05-01 10:27:19,729] {cli.py:71} INFO - Input dtype param  : float
[2026-05-01 10:27:19,729] {cli.py:72} INFO - Global rank        : 0
[2026-05-01 10:27:19,729] {cli.py:73} INFO - Local rank         : 0
[2026-05-01 10:27:19,729] {cli.py:74} INFO - ******************************
[2026-05-01 10:27:19,770] {commons.py:162} INFO - Python version: 3.10.12
[2026-05-01 10:27:19,770] {commons.py:163} INFO - PyTorch version: 2.9.0+cpu
[2026-05-01 10:27:19,770] {commons.py:164} INFO - PyTorch MPI enabled?: False
[2026-05-01 10:27:19,770] {commons.py:165} INFO - PyTorch CUDA enabled?: False
[2026-05-01 10:27:19,770] {commons.py:166} INFO - PyTorch NCCL enabled?: False
[2026-05-01 10:27:19,770] {commons.py:167} INFO - PyTorch Gloo enabled?: True
[2026-05-01 10:27:19,771] {commons.py:171} INFO - PyTorch HCCL enabled?: True
[2026-05-01 10:27:19,771] {commons.py:181} INFO - Using device *cpu* with *gloo* backend
[2026-05-01 10:27:19,771] {commons.py:182} INFO - World size: 2
[2026-05-01 10:27:19,771] {osu_util_mpi.py:31} INFO - # PyTorch Benchmark Latency Test
[2026-05-01 10:27:19,771] {osu_util_mpi.py:32} INFO - # Size (B) Elapsed Time (ms)
[2026-05-01 10:27:20,403] {osu_latency.py:72} INFO - 0                      28.63
[2026-05-01 10:27:21,211] {osu_latency.py:72} INFO - 4                      36.90
[2026-05-01 10:27:22,204] {osu_latency.py:72} INFO - 8                      45.06
[2026-05-01 10:27:23,197] {osu_latency.py:72} INFO - 16                     45.04
...
```

#### Results

Each benchmark writes a log file named after the benchmark (e.g. `latency.log`) and a CSV file
following the convention

```
osu_<benchmark>-<device>-<dtype>-<world_size>.csv
```

for example `osu_latency-npu-float16-2.csv`. The bandwidth-style benchmarks append the unit to the
name, e.g. `osu_bandwidth_gbps-npu-float16-2.csv`, `osu_bibw_gbps-...` and `osu_mbw_mr_gbps-...`.

### Plotting

`scripts/` contains sweep drivers and plotting utilities that turn the CSV files into PNG figures:

```bash
pip install -r requirements_plotting.txt

./scripts/benchmark_latency_bandwidth.sh   # latency/bandwidth sweep + plots
./scripts/benchmark_collectives.sh         # collectives over world sizes 2..8 + plots
```

Individual plotters (`scripts/plotter_latency.py`, `plotter_bandwidth.py`, `plotter_collectives.py`,
`plotter_mbw_mr.py`) read the CSVs of the current directory and respect the `DEVICE` and `HCCL_DTYPE`
environment variables.

### Known issues

* Gloo backend does not support `reduce_scatter`.

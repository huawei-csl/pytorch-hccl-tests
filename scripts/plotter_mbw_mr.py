#!/usr/bin/env python
"""Basic plotting utility for the multiple bandwidth / message rate benchmark.


Reads the per-world-size CSVs produced by the ``mbw_mr`` benchmark
(``osu_mbw_mr_gbps-<device>-<dtype>-<world_size>.csv``) and produces two
plots: aggregate bandwidth and aggregate message rate, both vs message size,
with one series per world size.

To generate the data, sweep world sizes, e.g.:

    for ws in 2 4 6 8; do
        WORLD_SIZE=$ws make mbw-mr DEVICE=npu
    done
    python scripts/plotter_mbw_mr.py


Requirements: `pip install matplotlib seaborn pandas`
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

plt.rcParams["lines.markersize"] = 20

sns.set(rc={"figure.figsize": (15.7, 8.27)})
sns.set_theme(style="ticks", palette="pastel")
sns.set_context("paper")
sns.set(font_scale=2)


DEVICE = os.environ.get("DEVICE", "cpu")
DTYPE = os.environ.get("HCCL_DTYPE", "float16")
PT_VER = torch.__version__

# WORLD_SIZES must be in sync with the world sizes swept when running mbw_mr.
WORLD_SIZES = [2, 3, 4, 5, 6, 7, 8]
col_name = "World Size"

X_LABEL = "size_in_bytes"

# Metric column -> (y-axis label, output filename tag)
METRICS = {
    "agg_bw_gbps": ("Aggregate Bandwidth (GB/s)", "bw"),
    "msg_rate_mmps": ("Aggregate Message Rate (M msgs/s)", "msgrate"),
}


def _plot(df, y_label, axis_label, tag):
    plt.figure()
    sns.despine(right=True)
    sns.lineplot(
        x=X_LABEL,
        y=y_label,
        hue=col_name,
        sizes=(20, 200),
        legend=False,
        data=df,
    )
    ax = sns.scatterplot(
        x=X_LABEL,
        y=y_label,
        hue=col_name,
        sizes=(20, 200),
        legend="auto",
        data=df,
    )

    ax.set(xscale="log")
    ax.set(yscale="log")
    ax.legend(markerscale=2)
    ax.set_xlabel("Message length (bytes)")
    ax.set_ylabel(axis_label)
    title = f"OSU-MPI mbw_mr benchmark\n (Device: {DEVICE} | dtype: {DTYPE}"
    title += f" | PT: {PT_VER}"
    title += ")"
    ax.set_title(title)

    plt.savefig(f"plot-mbw_mr-{tag}-{DEVICE}-{DTYPE}.png")
    plt.close()


def main():
    frames = []

    for world_size in WORLD_SIZES:
        s = str(world_size)
        path = f"osu_mbw_mr_gbps-{DEVICE}-{DTYPE}-{s}.csv"
        try:
            local = pd.read_csv(path)
            local[col_name] = s
            frames.append(local)
        except FileNotFoundError as err:
            print(f"Error: {err}")

    if not frames:
        print("No benchmark CSV files found; nothing to plot.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    df[col_name] = pd.Categorical(df[col_name])

    for column, (axis_label, tag) in METRICS.items():
        if column in df.columns:
            _plot(df, column, axis_label, tag)


if __name__ == "__main__":
    sys.exit(main())  # noqa

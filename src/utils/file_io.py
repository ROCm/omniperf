##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import os
import sys
import pandas as pd
import re
import yaml

import glob
import collections
from collections import OrderedDict
from pathlib import Path
from utils import schema
from utils.utils import console_debug, console_error, demarcate
from utils.kernel_name_shortener import kernel_name_shortener
import config

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd

# the build-in config to list kernel names purpose only
top_stats_build_in_config = {
    0: {
        "id": 0,
        "title": "Top Kernels",
        "data source": [{"raw_csv_table": {"id": 1, "source": "pmc_kernel_top.csv"}}],
    },
    1: {
        "id": 1,
        "title": "Dispatch List",
        "data source": [{"raw_csv_table": {"id": 2, "source": "pmc_dispatch_info.csv"}}],
    },
}

time_units = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def load_sys_info(f):
    """
    Load sys running info from csv file to a df.
    """
    return pd.read_csv(f)


def load_panel_configs(dir):
    """
    Load all panel configs from yaml file.
    """
    d = {}
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(".yaml"):
                with open(os.path.join(root, f)) as file:
                    config = yaml.safe_load(file)
                    d[config["Panel Config"]["id"]] = config["Panel Config"]

    # TODO: sort metrics as the header order in case they are not defined in the same order

    od = OrderedDict(sorted(d.items()))
    # for key, value in od.items():
    #     print(key, value)
    return od


@demarcate
def create_df_kernel_top_stats(
    raw_data_dir,
    filter_gpu_ids,
    filter_dispatch_ids,
    time_unit,
    max_stat_num,
    kernel_verbose,
    sortby="sum",
):
    """
    Create top stats info by grouping kernels with user's filters.
    """
    # NB:
    #   We even don't have to create pmc_kernel_top.csv explictly
    df = pd.read_csv(os.path.join(raw_data_dir, schema.pmc_perf_file_prefix + ".csv"))
    # Demangle original KernelNames
    kernel_name_shortener(df, kernel_verbose)

    # The logic below for filters are the same as in parser.apply_filters(),
    # which can be merged together if need it.
    if filter_gpu_ids:
        df = df.loc[df["GPU_ID"].astype(str).isin([filter_gpu_ids])]

    if filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        if ">" in filter_dispatch_ids[0]:
            m = re.match(r"\> (\d+)", filter_dispatch_ids[0])
            df = df[df["Dispatch_ID"] > int(m.group(1))]
        else:
            df = df.loc[df["Dispatch_ID"].astype(str).isin(filter_dispatch_ids)]

    # First, create a dispatches file used to populate global vars
    dispatch_info = df.loc[:, ["Dispatch_ID", "Kernel_Name", "GPU_ID"]]
    dispatch_info.to_csv(os.path.join(raw_data_dir, "pmc_dispatch_info.csv"), index=False)

    time_stats = pd.concat(
        [df["Kernel_Name"], (df["End_Timestamp"] - df["Start_Timestamp"])],
        keys=["Kernel_Name", "ExeTime"],
        axis=1,
    )

    grouped = time_stats.groupby(by=["Kernel_Name"]).agg(
        {"ExeTime": ["count", "sum", "mean", "median"]}
    )

    time_unit_str = "(" + time_unit + ")"
    grouped.columns = [
        x.capitalize() + time_unit_str if x != "count" else x.capitalize()
        for x in grouped.columns.get_level_values(1)
    ]

    key = "Sum" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])
    key = "Mean" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])
    key = "Median" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])

    grouped = grouped.reset_index()  # Remove special group indexing

    key = "Sum" + time_unit_str
    grouped["Pct"] = grouped[key] / grouped[key].sum() * 100

    # NB:
    #   Sort by total time as default.
    if sortby == "sum":
        grouped = grouped.sort_values(by=("Sum" + time_unit_str), ascending=False)
        grouped.to_csv(os.path.join(raw_data_dir, "pmc_kernel_top.csv"), index=False)
    elif sortby == "kernel":
        grouped = grouped.sort_values("Kernel_Name")
        grouped.to_csv(os.path.join(raw_data_dir, "pmc_kernel_top.csv"), index=False)


@demarcate
def create_df_pmc(raw_data_dir, kernel_verbose, verbose):
    """
    Load all raw pmc counters and join into one df.
    """
    dfs = []
    coll_levels = []

    df = pd.DataFrame()
    new_df = pd.DataFrame()
    for root, dirs, files in os.walk(raw_data_dir):
        for f in files:
            # print("file ", f)
            if (f.endswith(".csv") and f.startswith("SQ")) or (
                f == schema.pmc_perf_file_prefix + ".csv"
            ):
                tmp_df = pd.read_csv(os.path.join(root, f))
                # Demangle original KernelNames
                kernel_name_shortener(tmp_df, kernel_verbose)
                dfs.append(tmp_df)
                coll_levels.append(f[:-4])
    final_df = pd.concat(dfs, keys=coll_levels, axis=1, copy=False)
    if verbose >= 2:
        console_debug("pmc_raw_data final_df %s" % final_df.info)
    return final_df


def collect_wave_occu_per_cu(in_dir, out_dir, numSE):
    """
    Collect wave occupancy info from in_dir csv files
    and consolidate into out_dir/wave_occu_per_cu.csv.
    It depends highly on wave_occu_se*.csv format.
    """

    all = pd.DataFrame()

    for i in range(numSE):
        p = Path(in_dir, "wave_occu_se" + str(i) + ".csv")
        if p.exists():
            tmp_df = pd.read_csv(p)
            SE_idx = "SE" + str(tmp_df.loc[0, "SE"])
            tmp_df.rename(
                columns={
                    "Dispatch": "Dispatch",
                    "SE": "SE",
                    "CU": "CU",
                    "Occupancy": SE_idx,
                },
                inplace=True,
            )

            # TODO: join instead of concat!
            if i == 0:
                all = tmp_df[{"CU", SE_idx}]
                all.sort_index(axis=1, inplace=True)
            else:
                all = pd.concat([all, tmp_df[SE_idx]], axis=1, copy=False)

    if not all.empty:
        # print(all.transpose())
        all.to_csv(Path(out_dir, "wave_occu_per_cu.csv"), index=False)


def is_single_panel_config(root_dir, supported_archs):
    """
    Check the root configs dir structure to decide using one config set for all
    archs, or one for each arch.
    """
    # If not single config, verify all supported archs have defined configs
    supported_archs = supported_archs.keys()
    counter = 0
    for arch in supported_archs:
        if root_dir.joinpath(arch).exists():
            counter += 1
    if counter == 0:
        return True
    elif counter == len(supported_archs):
        return False
    else:
        console_error("Found multiple panel config sets but incomplete for all archs.")

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

from abc import ABC, abstractmethod
from tqdm import tqdm
import glob
import logging
import sys
import os
import re
from utils.utils import (
    capture_subprocess_output,
    run_prof,
    gen_sysinfo,
    run_rocscope,
    demarcate,
    console_log,
    console_debug,
    console_error,
    console_warning,
    print_status,
)
import config
import pandas as pd


class OmniProfiler_Base:
    def __init__(self, args, profiler_mode, soc):
        self.__args = args
        self.__profiler = profiler_mode
        self._soc = soc  # OmniSoC obj
        self.__perfmon_dir = os.path.join(
            str(config.omniperf_home), "omniperf_soc", "profile_configs"
        )

    def get_args(self):
        return self.__args

    def get_profiler_options(self, fname):
        """Fetch any version specific arguments required by profiler"""
        # assume no SoC specific options and return empty list by default
        return []

    @demarcate
    def pmc_perf_split(self):
        """Avoid default rocprof join utility by spliting each line into a separate input file"""
        workload_perfmon_dir = os.path.join(self.__args.path, "perfmon")
        lines = (
            open(os.path.join(workload_perfmon_dir, "pmc_perf.txt"), "r")
            .read()
            .splitlines()
        )

        # Iterate over each line in pmc_perf.txt
        mpattern = r"^pmc:(.*)"
        i = 0
        for line in lines:
            # Verify no comments
            stext = line.split("#")[0].strip()
            if not stext:
                continue

            # all pmc counters start with  "pmc:"
            m = re.match(mpattern, stext)
            if m is None:
                continue

            # Create separate file for each line
            fd = open(workload_perfmon_dir + "/pmc_perf_" + str(i) + ".txt", "w")
            fd.write(stext + "\n\n")
            fd.write("gpu:\n")
            fd.write("range:\n")
            fd.write("kernel:\n")
            fd.close()

            i += 1

        # Remove old pmc_perf.txt input from perfmon dir
        os.remove(workload_perfmon_dir + "/pmc_perf.txt")

    @demarcate
    def join_prof(self, out=None):
        """Manually join separated rocprof runs"""
        # Set default output directory if not specified
        if type(self.__args.path) == str:
            if out is None:
                out = self.__args.path + "/pmc_perf.csv"
            files = glob.glob(self.__args.path + "/" + "pmc_perf_*.csv")
        elif type(self.__args.path) == list:
            files = self.__args.path
        else:
            console_error(
                "Invalid workload directory. Cannot resolve %s" % self.__args.path
            )

        df = None
        for i, file in enumerate(files):
            _df = pd.read_csv(file) if type(self.__args.path) == str else file
            if self.__args.join_type == "kernel":
                key = _df.groupby("Kernel_Name").cumcount()
                _df["key"] = _df.Kernel_Name + " - " + key.astype(str)
            elif self.__args.join_type == "grid":
                key = _df.groupby(["Kernel_Name", "Grid_Size"]).cumcount()
                _df["key"] = (
                    _df["Kernel_Name"]
                    + " - "
                    + _df["Grid_Size"].astype(str)
                    + " - "
                    + key.astype(str)
                )
            else:
                console_error(
                    "%s is an unrecognized option for --join-type" % self.__args.join_type
                )

            if df is None:
                df = _df
            else:
                # join by unique index of kernel
                df = pd.merge(df, _df, how="inner", on="key", suffixes=("", f"_{i}"))

        # TODO: check for any mismatch in joins
        duplicate_cols = {
            "GPU_ID": [col for col in df.columns if col.startswith("GPU_ID")],
            "Grid_Size": [col for col in df.columns if col.startswith("Grid_Size")],
            "Workgroup_Size": [
                col for col in df.columns if col.startswith("Workgroup_Size")
            ],
            "LDS_Per_Workgroup": [
                col for col in df.columns if col.startswith("LDS_Per_Workgroup")
            ],
            "Scratch_Per_Workitem": [
                col for col in df.columns if col.startswith("Scratch_Per_Workitem")
            ],
            "SGPR": [col for col in df.columns if col.startswith("SGPR")],
        }
        # Check for vgpr counter in ROCm < 5.3
        if "vgpr" in df.columns:
            duplicate_cols["vgpr"] = [col for col in df.columns if col.startswith("vgpr")]
        # Check for vgpr counter in ROCm >= 5.3
        else:
            duplicate_cols["Arch_VGPR"] = [
                col for col in df.columns if col.startswith("Arch_VGPR")
            ]
            duplicate_cols["Accum_VGPR"] = [
                col for col in df.columns if col.startswith("Accum_VGPR")
            ]
        for key, cols in duplicate_cols.items():
            _df = df[cols]
            if not test_df_column_equality(_df):
                msg = "Detected differing {} values while joining pmc_perf.csv".format(
                    key
                )
                console_warning(msg)
            else:
                msg = "Successfully joined {} in pmc_perf.csv".format(key)
                console_debug(msg)

        # now, we can:
        #   A) throw away any of the "boring" duplicates
        df = df[
            [
                k
                for k in df.keys()
                if not any(
                    k.startswith(check)
                    for check in [
                        # rocprofv2 headers
                        "GPU_ID_",
                        "Grid_Size_",
                        "Workgroup_Size_",
                        "LDS_Per_Workgroup_",
                        "Scratch_Per_Workitem_",
                        "vgpr_",
                        "Arch_VGPR_",
                        "Accum_VGPR_",
                        "SGPR_",
                        "Dispatch_ID_",
                        "Queue_ID",
                        "Queue_Index",
                        "PID",
                        "TID",
                        "SIG",
                        "OBJ",
                        # rocscope specific merged counters, keep original
                        "dispatch_",
                        # extras
                        "sig",
                        "queue-id",
                        "queue-index",
                        "pid",
                        "tid",
                        "fbar",
                    ]
                )
            ]
        ]
        #   B) any timestamps that are _not_ the duration, which is the one we care about
        df = df[
            [
                k
                for k in df.keys()
                if not any(
                    check in k
                    for check in [
                        "DispatchNs",
                        "CompleteNs",
                        # rocscope specific timestamp
                        "HostDuration",
                    ]
                )
            ]
        ]
        #   C) sanity check the name and key
        namekeys = [k for k in df.keys() if "Kernel_Name" in k]
        assert len(namekeys)
        for k in namekeys[1:]:
            assert (df[namekeys[0]] == df[k]).all()
        df = df.drop(columns=namekeys[1:])
        # now take the median of the durations
        bkeys = []
        ekeys = []
        for k in df.keys():
            if "Start_Timestamp" in k:
                bkeys.append(k)
            if "End_Timestamp" in k:
                ekeys.append(k)
        # compute mean begin and end timestamps
        endNs = df[ekeys].mean(axis=1)
        beginNs = df[bkeys].mean(axis=1)
        # and replace
        df = df.drop(columns=bkeys)
        df = df.drop(columns=ekeys)
        df["Start_Timestamp"] = beginNs
        df["End_Timestamp"] = endNs
        # finally, join the drop key
        df = df.drop(columns=["key"])
        # save to file and delete old file(s), skip if we're being called outside of Omniperf
        if type(self.__args.path) == str:
            df.to_csv(out, index=False)
            if not self.__args.verbose:
                for file in files:
                    os.remove(file)
        else:
            return df

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling."""
        console_debug("profiling", "pre-processing using %s profiler" % self.__profiler)

        # verify soc compatibility
        if self.__profiler not in self._soc.get_compatible_profilers():
            console_error(
                "%s is not enabled in %s. Available profilers include: %s"
                % (
                    self._soc.get_arch(),
                    self.__profiler,
                    self._soc.get_compatible_profilers(),
                )
            )
        # verify not accessing parent directories
        if ".." in str(self.__args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
            )

        # verify correct formatting for application binary
        self.__args.remaining = self.__args.remaining[1:]
        if self.__args.remaining:
            if not os.path.isfile(self.__args.remaining[0]):
                console_error(
                    "Your command %s doesn't point to a executable. Please verify."
                    % self.__args.remaining[0]
                )
            self.__args.remaining = " ".join(self.__args.remaining)
        else:
            console_error(
                "Profiling command required. Pass application executable after -- at the end of options.\n\t\ti.e. omniperf profile -n vcopy -- ./vcopy 1048576 256"
            )

        # verify name meets MongoDB length requirements and no illegal chars
        if len(self.__args.name) > 35:
            console_error("-n/--name exceeds 35 character limit. Try again.")
        if self.__args.name.find(".") != -1 or self.__args.name.find("-") != -1:
            console_error("'-' and '.' are not permitted in -n/--name")

    @abstractmethod
    def run_profiling(self, version: str, prog: str):
        """Run profiling."""
        console_debug(
            "profiling", "performing profiling using %s profiler" % self.__profiler
        )

        # log basic info
        console_log(str(prog).title() + " version: " + str(version))
        console_log("Profiler choice: %s" % self.__profiler)
        console_log("Path: " + str(os.path.abspath(self.__args.path)))
        console_log("Target: " + str(self._soc._mspec.gpu_model))
        console_log("Command: " + str(self.__args.remaining))
        console_log("Kernel Selection: " + str(self.__args.kernel))
        console_log("Dispatch Selection: " + str(self.__args.dispatch))
        if self.__args.ipblocks == None:
            console_log("Hardware Blocks: All")
        else:
            console_log("Hardware Blocks: " + str(self.__args.ipblocks))

        print_status("Collecting Performance Counters")

        # show status bar in error-only mode
        disable_tqdm = True
        if self.__args.loglevel >= logging.ERROR:
            disable_tqdm = False

        # Run profiling on each input file
        input_files = glob.glob(self.get_args().path + "/perfmon/*.txt")
        input_files.sort()

        for fname in tqdm(input_files, disable=disable_tqdm):
            # Kernel filtering (in-place replacement)
            if not self.__args.kernel == None:
                success, output = capture_subprocess_output(
                    [
                        "sed",
                        "-i",
                        "-r",
                        "s%^(kernel:).*%"
                        + "kernel: "
                        + ",".join(self.__args.kernel)
                        + "%g",
                        fname,
                    ]
                )
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)

            # Dispatch filtering (inplace replacement)
            if not self.__args.dispatch == None:
                success, output = capture_subprocess_output(
                    [
                        "sed",
                        "-i",
                        "-r",
                        "s%^(range:).*%"
                        + "range: "
                        + " ".join(self.__args.dispatch)
                        + "%g",
                        fname,
                    ]
                )
                # log output from profile filtering
                if not success:
                    console_error(output)
                else:
                    console_debug(output)
            console_log("profiling", "Current input file: %s" % fname)

            # Fetch any SoC/profiler specific profiling options
            options = self._soc.get_profiler_options()
            options += self.get_profiler_options(fname)
            if self.__profiler == "rocprofv1" or self.__profiler == "rocprofv2":
                run_prof(
                    fname=fname,
                    profiler_options=options,
                    workload_dir=self.get_args().path,
                    mspec=self._soc._mspec,
                    loglevel=self.get_args().loglevel,
                )

            elif self.__profiler == "rocscope":
                run_rocscope(self.__args, fname)
            else:
                # TODO: Finish logic
                console_error("Profiler not supported")

    @abstractmethod
    def post_processing(self):
        """Perform any post-processing steps prior to profiling."""
        console_debug(
            "profiling", "performing post-processing using %s profiler" % self.__profiler
        )

        gen_sysinfo(
            workload_name=self.__args.name,
            workload_dir=self.get_args().path,
            ip_blocks=self.__args.ipblocks,
            app_cmd=self.__args.remaining,
            skip_roof=self.__args.no_roof,
            roof_only=self.__args.roof_only,
            mspec=self._soc._mspec,
            soc=self._soc,
        )


def test_df_column_equality(df):
    return df.eq(df.iloc[:, 0], axis=0).all(1).all()

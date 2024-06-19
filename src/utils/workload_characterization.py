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
import re
import csv
import pandas as pd
import numpy as np
from perfetto.trace_processor import TraceProcessor
from utils.utils import console_error, console_log, console_debug
from utils.plot_characterization import create_plots


class Bottleneck_Classification:
    def __init__(self, omniperf_dir, omnitrace_dir, treshold_ratio):
        self.input_dirs = {"omniperf": omniperf_dir, "omnitrace": omnitrace_dir}
        self.util_ratio = treshold_ratio
        # Initalize data structures
        self.omnitrace_data = {
            "gpu_ids": [],
            "total_time": 0,
            "gpu_time": {},
            "host_device_time": {},
            "device_time_host": {},
            "gpu_gpu_comm_time": {},
            "gpu_invoke_time": {},
        }
        self.omniperf_data = {"gpu_bounds_time": {}, "gpu_gpu_comm_time": {}}
        self.roofline_max_map = {}
        # Populate data structures
        self.parse_omniperf()
        self.parse_omnitrace()

    def parse_omniperf(self):
        """
        The utilization ratio is used to set a threshold for percentage of the empirical peak FLOPS/s.
        Used to classify a kernel as performing well and underperforming.
        """
        perf_df = pd.read_csv(
            os.path.join(self.input_dirs["omniperf"][0][0], "pmc_perf.csv")
        )
        # Verify that roofline.csv exists
        if not os.path.exists(
            os.path.abspath(
                os.path.join(self.input_dirs["omniperf"][0][0], "roofline.csv")
            )
        ):
            console_error(
                f"The Omniperf file {os.path.join(self.input_dirs['omniperf'][0][0], 'roofline.csv')} is required to use the --bottleneck-trace flag."
            )
        roofline_df = pd.read_csv(
            os.path.join(self.input_dirs["omniperf"][0][0], "roofline.csv")
        )

        # go row by row in pmc_perf to parse the relevant performance counters per kernel to
        # collect the parameters needed for hierarchical rooflines by calling the helper functions
        # kernel data map uses gpu-id as the key, and maps to a list of data tuples
        kernel_data_map = {}
        for i, row in perf_df.iterrows():
            gpu_id = row["GPU_ID"]
            curr_kernel_name = row["Kernel_Name"]
            curr_kernel_dur = row["End_Timestamp"] - row["Start_Timestamp"]

            valu_num_ops, mfma_num_ops = find_ops_bytes_mi200(row)
            data_row = (
                curr_kernel_name,
                curr_kernel_dur,
                valu_num_ops,
                mfma_num_ops,
                find_lds_bytes(row),
                find_l1d_bytes(row),
                find_l2_bytes(row),
                find_hbm_bytes(row),
            )

            kernel_data_map.setdefault(gpu_id, []).append(data_row)

        self.parse_roofline_data(roofline_df, kernel_data_map.keys())

        # lds bound time, l1d time, l2 time, hbm time, valu, mfma compute bound time
        bottleneck_classified_time_map = {}

        # rccl nccl
        gpu_gpu_comm_time_map = {}

        # for each kernel, classify it by it's bandwidth/compute bound bucket
        # and also by it's flops/sec performance by if it either uses
        # no flops, under util_ratio's percent of peak, or above util_ratio's threshold

        # track data separately for each GPU in the system
        for gpu_id, data_tups in kernel_data_map.items():
            # these arrays' indicies correspond to lds, gl1, gl2, hbm, valu, mfma
            above_threshold_time = [0, 0, 0, 0, 0, 0]
            below_threshold_time = [0, 0, 0, 0, 0, 0]
            no_flops_time = [0, 0, 0, 0, 0, 0]

            (
                peak_bw_lds,
                peak_bw_l1d,
                peak_bw_l2,
                peak_bw_hbm,
                peak_valu_flops,
                peak_mfma_flops,
            ) = self.roofline_max_map[gpu_id]
            gpu_gpu_comm_time_map[gpu_id] = 0

            for tups in data_tups:
                (
                    kernel_name,
                    kernel_dur,
                    valu_num_ops,
                    mfma_num_ops,
                    num_lds_bytes,
                    num_l1d_bytes,
                    num_l2_bytes,
                    num_hbm_bytes,
                ) = tups

                # Similar to the code for Omnitrace earlier, ignore fillbuffer calls
                if "fillbuffer" in kernel_name.lower():
                    continue
                # we don't classify rccl calls in the gpu to prevent
                # double counting gpu-gpu communication time
                elif "rccl" in kernel_name.lower() or "nccl" in kernel_name.lower():
                    gpu_gpu_comm_time_map[gpu_id] += kernel_dur
                    continue

                # Each of these lines is used to help calculate a kernel's bottleneck  (bandwidth or compute) at the 4
                # memory levels (LDS, L1, L2, and HBM). These values will be used to classify the bottlenecks later
                # in the code.
                hbm_bound, hbm_pop, hbm_ai = calc_bottleneck(
                    valu_num_ops,
                    mfma_num_ops,
                    peak_valu_flops,
                    peak_mfma_flops,
                    num_hbm_bytes,
                    peak_bw_hbm,
                    kernel_dur,
                )
                l2_bound, l2_pop, l2_ai = calc_bottleneck(
                    valu_num_ops,
                    mfma_num_ops,
                    peak_valu_flops,
                    peak_mfma_flops,
                    num_l2_bytes,
                    peak_bw_l2,
                    kernel_dur,
                )
                l1d_bound, l1d_pop, l1d_ai = calc_bottleneck(
                    valu_num_ops,
                    mfma_num_ops,
                    peak_valu_flops,
                    peak_mfma_flops,
                    num_l1d_bytes,
                    peak_bw_l1d,
                    kernel_dur,
                )
                lds_bound, lds_pop, lds_ai = calc_bottleneck(
                    valu_num_ops,
                    mfma_num_ops,
                    peak_valu_flops,
                    peak_mfma_flops,
                    num_lds_bytes,
                    peak_bw_lds,
                    kernel_dur,
                )

                # The bottlneck classification has been heavily modified since August's initial implementation.
                # We first compare the number of HBM bytes, MFMA operations, and VALU operations to see which occurred
                # most for the kernel. Based on that, we then enter a set of conditionals that prioritize based on which
                # occurred most. If there was a lot of HBM traffic, we enter the first set of conditionals that prioritizes
                # HBM and starts there and moves through the memory heirachy and then go to compute.
                # If there were more MFMA or VALU operations, we enter different sets of conditionals that prioritize
                # MFMA or VALU and then moves through the memory heirachy. Through validation it was discovered that
                # the order in which these conditionals occurred greatly changed the bottlenecks. This configuration yielded
                # the correct and expected results. If modifications are required to prioritize the LDS, L1, or L2, that
                # can be done.
                if num_hbm_bytes > mfma_num_ops and num_hbm_bytes > valu_num_ops:
                    # Used to debug
                    console_debug(
                        "Bottleneck Classifier",
                        f"I am in the HBM conditional -> num_hbm_bytes: {num_hbm_bytes} mfma_num_ops: {mfma_num_ops} valu_num_ops: {valu_num_ops}",
                    )
                    if hbm_bound == "bandwidth":
                        if hbm_ai == 0:
                            no_flops_time[3] += kernel_dur
                        # How close was this kernel to the peak and is that below the set utilization ratio?
                        elif hbm_pop < self.util_ratio:
                            below_threshold_time[3] += kernel_dur
                        else:
                            above_threshold_time[3] += kernel_dur
                    elif l2_bound == "bandwidth":
                        if l2_ai == 0:
                            no_flops_time[2] += kernel_dur
                        elif l2_pop < self.util_ratio:
                            below_threshold_time[2] += kernel_dur
                        else:
                            above_threshold_time[2] += kernel_dur
                    elif l1d_bound == "bandwidth":
                        if l1d_ai == 0:
                            no_flops_time[1] += kernel_dur
                        elif l1d_pop < self.util_ratio:
                            below_threshold_time[1] += kernel_dur
                        else:
                            above_threshold_time[1] += kernel_dur
                    elif lds_bound == "bandwidth":
                        if lds_ai == 0:
                            no_flops_time[0] += kernel_dur
                        elif lds_pop < self.util_ratio:
                            below_threshold_time[0] += kernel_dur
                        else:
                            above_threshold_time[0] += kernel_dur
                    elif valu_num_ops > mfma_num_ops:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[4] += kernel_dur
                        else:
                            above_threshold_time[4] += kernel_dur
                    else:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[5] += kernel_dur
                        else:
                            above_threshold_time[5] += kernel_dur
                elif mfma_num_ops > num_hbm_bytes and mfma_num_ops > valu_num_ops:
                    console_debug(
                        "Bottleneck Classifier",
                        f"I am in the MFMA conditional -> num_hbm_bytes: {num_hbm_bytes} mfma_num_ops: {mfma_num_ops} valu_num_ops: {valu_num_ops}",
                    )
                    if mfma_num_ops > valu_num_ops:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[5] += kernel_dur
                        else:
                            above_threshold_time[5] += kernel_dur
                    elif hbm_bound == "bandwidth":
                        if hbm_ai == 0:
                            no_flops_time[3] += kernel_dur
                        elif hbm_pop < self.util_ratio:
                            below_threshold_time[3] += kernel_dur
                        else:
                            above_threshold_time[3] += kernel_dur
                    elif l2_bound == "bandwidth":
                        if l2_ai == 0:
                            no_flops_time[2] += kernel_dur
                        elif l2_pop < self.util_ratio:
                            below_threshold_time[2] += kernel_dur
                        else:
                            above_threshold_time[2] += kernel_dur
                    elif l1d_bound == "bandwidth":
                        if l1d_ai == 0:
                            no_flops_time[1] += kernel_dur
                        elif l1d_pop < self.util_ratio:
                            below_threshold_time[1] += kernel_dur
                        else:
                            above_threshold_time[1] += kernel_dur
                    elif lds_bound == "bandwidth":
                        if lds_ai == 0:
                            no_flops_time[0] += kernel_dur
                        elif lds_pop < self.util_ratio:
                            below_threshold_time[0] += kernel_dur
                        else:
                            above_threshold_time[0] += kernel_dur
                    else:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[4] += kernel_dur
                        else:
                            above_threshold_time[4] += kernel_dur
                elif valu_num_ops > num_hbm_bytes and valu_num_ops > mfma_num_ops:
                    console_debug(
                        "Bottleneck Classifier",
                        "I am in the VALU conditional -> num_hbm_bytes: {num_hbm_bytes} mfma_num_ops: {mfma_num_ops} valu_num_ops: {valu_num_ops}",
                    )
                    if valu_num_ops > mfma_num_ops:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[4] += kernel_dur
                        else:
                            above_threshold_time[4] += kernel_dur
                    elif hbm_bound == "bandwidth":
                        if hbm_ai == 0:
                            no_flops_time[3] += kernel_dur
                        elif hbm_pop < self.util_ratio:
                            below_threshold_time[3] += kernel_dur
                        else:
                            above_threshold_time[3] += kernel_dur
                    elif l2_bound == "bandwidth":
                        if l2_ai == 0:
                            no_flops_time[2] += kernel_dur
                        elif l2_pop < self.util_ratio:
                            below_threshold_time[2] += kernel_dur
                        else:
                            above_threshold_time[2] += kernel_dur
                    elif l1d_bound == "bandwidth":
                        if l1d_ai == 0:
                            no_flops_time[1] += kernel_dur
                        elif l1d_pop < self.util_ratio:
                            below_threshold_time[1] += kernel_dur
                        else:
                            above_threshold_time[1] += kernel_dur
                    elif lds_bound == "bandwidth":
                        if lds_ai == 0:
                            no_flops_time[0] += kernel_dur
                        elif lds_pop < self.util_ratio:
                            below_threshold_time[0] += kernel_dur
                        else:
                            above_threshold_time[0] += kernel_dur
                    else:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[5] += kernel_dur
                        else:
                            above_threshold_time[5] += kernel_dur
                else:
                    if hbm_bound == "bandwidth":
                        if hbm_ai == 0:
                            no_flops_time[3] += kernel_dur
                        elif hbm_pop < self.util_ratio:
                            below_threshold_time[3] += kernel_dur
                        else:
                            above_threshold_time[3] += kernel_dur
                    elif l2_bound == "bandwidth":
                        if l2_ai == 0:
                            no_flops_time[2] += kernel_dur
                        elif l2_pop < self.util_ratio:
                            below_threshold_time[2] += kernel_dur
                        else:
                            above_threshold_time[2] += kernel_dur
                    elif l1d_bound == "bandwidth":
                        if l1d_ai == 0:
                            no_flops_time[1] += kernel_dur
                        elif l1d_pop < self.util_ratio:
                            below_threshold_time[1] += kernel_dur
                        else:
                            above_threshold_time[1] += kernel_dur
                    elif lds_bound == "bandwidth":
                        if lds_ai == 0:
                            no_flops_time[0] += kernel_dur
                        elif lds_pop < self.util_ratio:
                            below_threshold_time[0] += kernel_dur
                        else:
                            above_threshold_time[0] += kernel_dur
                    elif valu_num_ops > mfma_num_ops:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[4] += kernel_dur
                        else:
                            above_threshold_time[4] += kernel_dur
                    else:
                        if hbm_pop < self.util_ratio:
                            below_threshold_time[5] += kernel_dur
                        else:
                            above_threshold_time[5] += kernel_dur

            bottleneck_classified_time = [
                no_flops_time,
                below_threshold_time,
                above_threshold_time,
            ]
            bottleneck_classified_time_map[gpu_id] = bottleneck_classified_time

        self.omniperf_data["gpu_bounds_time"] = bottleneck_classified_time_map
        self.omniperf_data["gpu_gpu_comm_time"] = gpu_gpu_comm_time_map

    def parse_roofline_data(self, roof_df, gpu_ids):
        """
        Sets map of gpu_ids and corresponding roofline maxes
        """
        roofline_data_map = {}

        # the device id in roofline.csv (0, 1) does not match the gpu ids in pmc_perf.csv (2, 3)
        # so use two separate counters
        roofline_counter = 0
        for id in gpu_ids:
            peak_bw_lds = roof_df["LDSBw"][roofline_counter]
            peak_bw_l1d = roof_df["L1Bw"][roofline_counter]
            peak_bw_l2 = roof_df["L2Bw"][roofline_counter]
            peak_bw_hbm = roof_df["HBMBw"][roofline_counter]
            # These two values are the peak valu and mfma performance for a certain data type.
            # They need to be changed if you are using a different data type.
            # For VALU, The options are: FP64Flops and FP32Flops
            # For MFMA, the options are: MFMAF64Flops, MFMAF32Flops, MFMAF16Flops, MFMABF16Flops
            peak_valu_flops = roof_df["FP32Flops"][roofline_counter]
            peak_mfma_flops = roof_df["MFMAF32Flops"][roofline_counter]

            roofline_data_map[id] = (
                peak_bw_lds,
                peak_bw_l1d,
                peak_bw_l2,
                peak_bw_hbm,
                peak_valu_flops,
                peak_mfma_flops,
            )

            roofline_counter += 1

        self.roofline_max_map = roofline_data_map

    def parse_omnitrace(self):
        """
        Buckets for omnitrace time analysis:
        cpu-gpu-mem:            time where spent where cpu and gpu are copy/setting memory between each other this is further broken down by the direction (cpu to gpu, gpu to cpu) and gpu-gpu communication
        gpu-kernel:             time gpu is spent running work that is invoked by the cpu by `hipLaunchKernel`
        kernel-invoke-overhead: gpu kernel invocation overhead (time after `hipLaunchKernel` ends on the CPU and the corresponding GPU kernel starts running
        cpu:                    any time other time that traced by omnitrace
        """
        if not os.path.exists(os.path.abspath(self.input_dirs["omnitrace"])):
            console_error(
                f"The Omnitrace file {self.input_dirs['omnitrace']} cannot be found."
            )
        tp = open_proto_file(os.path.abspath(self.input_dirs["omnitrace"]))

        # this large query results a SQL table where the caller is the CPU function that launches
        # a kernel, and the callee is the corresponding GPU kernel that is invoked
        # All of these commands can also be input into the Perfetto UI manually to see the information. Good for debugging.
        trace_stats_query = "SELECT caller_slice.name AS caller_name, \
                            caller_slice.ts AS caller_ts, \
                            caller_slice.dur AS caller_dur, \
                            caller_slice.id AS caller_id, \
                            caller_slice.name AS caller_name, \
                            caller_slice.track_id AS caller_track_id, \
                            callee_slice.ts AS callee_ts, \
                            callee_slice.dur AS callee_dur, \
                            callee_slice.id AS callee_id, \
                            callee_slice.name AS callee_name, \
                            callee_slice.track_id AS callee_track_id \
                            FROM (slice AS caller_slice JOIN flow ON caller_slice.id = flow.slice_out) \
                            JOIN slice AS callee_slice ON callee_slice.id = flow.slice_in"

        trace_stats_iter = tp.query(trace_stats_query)
        gpu_tracks_map = parse_gpu_track_map(tp)

        console_log("The trace stats iter is: ", trace_stats_iter)
        console_log("The gpu tracks map is: ", gpu_tracks_map)

        # for multi gpu, classify the rows/slices by which GPU they are invoked on
        # GPU kernels for a single GPU may show up on multiple GPU tracks
        gpu_slices_map = {}
        for row in trace_stats_iter:
            for gpu_id in gpu_tracks_map.keys():
                if row.callee_track_id in gpu_tracks_map[gpu_id]:
                    gpu_slices_map.setdefault(gpu_id, []).append(row)
                    # console_log('I am appending with the row: ', row)

        # we use the CPU's function name to determine the type of workload being invoked on the GPU
        # this is since the gpu kernel's names are intractable to parse
        # these are common substrings in the Proto file for kernel and memory communication
        launch_kernel_substring = ["launchkernel"]
        cpu_gpu_mem_substrings = ["memcpy", "memset", "malloc"]

        # keep track of time of each occurance. useful for summary stats
        # for multi gpu, this is done per gpu
        device_to_host_mem_map = {}
        host_to_device_mem_map = {}
        gpu_gpu_comm_map = {}
        gpu_kernel_map = {}

        # use to determine if a kernel is launched by the cpu when the GPU is running another kernel
        gpu_kernel_cold_invoke_overhead_map = {}
        gpu_kernel_stall_invoke_delay_map = {}

        # this keeps track time intervals when something is running on the GPU
        # and used for cold invoke times
        merged_kernel_slices_map = {}
        for gpu_id, gpu_tracks in gpu_tracks_map.items():
            merged_kernel_slices_map[gpu_id] = parse_merge_multi_gpu_slices(
                tp, gpu_tracks
            )

        for gpu_id in gpu_tracks_map.keys():
            for row in gpu_slices_map[gpu_id]:
                curr_gpu_kernel_i = 0
                # Look for any of the substrings above so we can accumulate those run times.
                if any(
                    substring in row.caller_name.lower()
                    for substring in cpu_gpu_mem_substrings
                ):
                    # for memory ops, the gpu memory kernel may start before the cpu's function ends (there's overlap)
                    # avoid double counting the overlap
                    mem_op_time = (row.callee_ts + row.callee_dur) - row.caller_ts
                    if row.callee_name.lower() == "copydevicetodevice":
                        gpu_gpu_comm_map.setdefault(gpu_id, []).append(row.callee_dur)
                    elif row.callee_name.lower() == "copydevicetohost":
                        device_to_host_mem_map.setdefault(gpu_id, []).append(mem_op_time)
                    elif (
                        row.callee_name.lower() == "copyhosttodevice"
                        or row.caller_name.lower() == "hipmemset"
                    ):
                        host_to_device_mem_map.setdefault(gpu_id, []).append(mem_op_time)

                    # fillbuffer calls are controversial since some are inherent in the
                    # program, but are also often inserted by ROCM profiler (and these calls would
                    # not show up when running without omnitrace/omniperf). We ignore them since
                    # they do not significantly affect runtime
                    elif row.callee_name.lower() == "fillbuffer":
                        continue
                    else:
                        raise Exception("not classified")

                # case when kernel is launched by the cpu via hiplaunchkernel
                elif any(
                    substring in row.caller_name.lower()
                    for substring in launch_kernel_substring
                ):
                    # r/nccl calls represent GPU to GPU communications but are launched by the CPU
                    if ("nccl" in row.callee_name.lower()) or (
                        "rccl" in row.callee_name.lower()
                    ):
                        gpu_gpu_comm_map.setdefault(gpu_id, []).append(row.callee_dur)
                    else:
                        # all other calls we classify as compute
                        gpu_kernel_map.setdefault(gpu_id, []).append(row.callee_dur)

                    invoke_time = calc_delay(row.caller_ts, row.caller_dur, row.callee_ts)

                    # find the relevant slice where the gpu kernel's end time is greater than
                    # the cpu's launchkernel's slice end time
                    while merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][1] < (
                        row.caller_ts + row.caller_dur
                    ):
                        curr_gpu_kernel_i += 1

                    # if the launchkernel call ended while a gpu slice is running, this is a stalled invoke
                    # otherwise this is a cold invoke
                    if (
                        merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][0]
                        <= (row.caller_ts + row.caller_dur)
                        and (row.caller_ts + row.caller_dur)
                        <= merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][1]
                    ):
                        gpu_kernel_stall_invoke_delay_map.setdefault(gpu_id, []).append(
                            invoke_time
                        )
                    else:
                        # nothing is running on the GPU, so this is a cold invoke
                        # sanity check: for this case the happen, the launchkernel end time must happen
                        # before the next kernel's start time
                        assert (
                            row.caller_ts + row.caller_dur
                            <= merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][0]
                        )
                        gpu_kernel_cold_invoke_overhead_map.setdefault(gpu_id, []).append(
                            invoke_time
                        )

            ## if your omnitrace database has bugs and your GPU time is being undercounted,
            ## you can choose to return gpu_busy_time_map istead of gpu_kernel_map
            # gpu_busy_time_map = parse_gpu_busy(tp, gpu_tracks_map.keys())

        total_traced_runtime = parse_top_process_functions_time(tp)

        self.omnitrace_data["gpu_ids"] = gpu_tracks_map.keys()
        self.omnitrace_data["total_time"] = total_traced_runtime
        self.omnitrace_data["gpu_time"] = gpu_kernel_map
        self.omnitrace_data["host_device_time"] = host_to_device_mem_map
        self.omnitrace_data["device_time_host"] = device_to_host_mem_map
        self.omnitrace_data["gpu_gpu_comm_time"] = gpu_gpu_comm_map
        self.omnitrace_data["gpu_invoke_time"] = gpu_kernel_cold_invoke_overhead_map

    def create_output_plots(self):
        # Organize output data into a dataframe for easy plotting
        output_headers = [
            "gpu_id",
            "ot_total_trace_time",
            "ot_gpu_time",
            "ot_host_device_time",
            "ot_device_host_time",
            "ot_gpu_gpu_comm_time",
            "ot_invoke_time",
            "op_gpu_time",
            "op_gpu_gpu_comm_time",
            "ot_util_threshold",
            "ot_no_flops_lds_time",
            "ot_no_flops_gl1_time",
            "ot_no_flops_gl2_time",
            "ot_no_flops_hbm_time",
            "ot_no_flops_valu_time",
            "ot_no_flops_mfma_time",
            "ot_under_util_lds_time",
            "ot_under_util_gl1_time",
            "ot_under_util_gl2_time",
            "ot_under_util_hbm_time",
            "ot_under_util_valu_time",
            "ot_under_util_mfma_time",
            "ot_above_util_lds_time",
            "ot_above_util_gl1_time",
            "ot_above_util_gl2_time",
            "ot_above_util_hbm_time",
            "ot_above_util_valu_time",
            "ot_above_util_mfma_time",
        ]
        output_df = pd.DataFrame(columns=output_headers)
        for gpu_id in self.omnitrace_data["gpu_ids"]:
            output_row = []
            output_row.append(gpu_id)
            output_row.append(self.omnitrace_data["total_time"])
            if self.omnitrace_data["gpu_time"]:
                output_row.append(sum(self.omnitrace_data["gpu_time"][gpu_id]))
            else:
                output_row.append(0)
            if self.omnitrace_data["host_device_time"]:
                output_row.append(sum(self.omnitrace_data["host_device_time"][gpu_id]))
            else:
                output_row.append(0)
            if self.omnitrace_data["device_time_host"]:
                output_row.append(sum(self.omnitrace_data["device_time_host"][gpu_id]))
            else:
                output_row.append(0)
            if self.omnitrace_data["gpu_gpu_comm_time"]:
                output_row.append(sum(self.omnitrace_data["gpu_gpu_comm_time"][gpu_id]))
            else:
                output_row.append(0)
            if self.omnitrace_data["gpu_invoke_time"]:
                output_row.append(sum(self.omnitrace_data["gpu_invoke_time"][gpu_id]))
            else:
                output_row.append(0)
            op_no_flops_time, op_below_util_time, op_above_util_time = self.omniperf_data[
                "gpu_bounds_time"
            ][gpu_id]
            op_gpu_sum = (
                self.omniperf_data["gpu_gpu_comm_time"][gpu_id]
                + sum(op_no_flops_time)
                + sum(op_below_util_time)
                + sum(op_above_util_time)
            )
            output_row.append(op_gpu_sum)
            output_row.append(self.omniperf_data["gpu_gpu_comm_time"][gpu_id])
            output_row.append(self.util_ratio)
            output_row.extend(op_no_flops_time)
            output_row.extend(op_below_util_time)
            output_row.extend(op_above_util_time)
            output_df.loc[len(output_df)] = output_row

        # Share df with plot creation module
        return create_plots(output_df, self.input_dirs["omniperf"][0][0])


def find_ops_bytes_mi200(df_row):
    valu_factor = 64
    mfma_factor = 512

    valu_fops = (
        valu_factor
        * (
            df_row["SQ_INSTS_VALU_ADD_F16"]
            + df_row["SQ_INSTS_VALU_MUL_F16"]
            + 2 * df_row["SQ_INSTS_VALU_FMA_F16"]
            + df_row["SQ_INSTS_VALU_TRANS_F16"]
        )
        + valu_factor
        * (
            df_row["SQ_INSTS_VALU_ADD_F32"]
            + df_row["SQ_INSTS_VALU_MUL_F32"]
            + 2 * df_row["SQ_INSTS_VALU_FMA_F32"]
            + df_row["SQ_INSTS_VALU_TRANS_F32"]
        )
        + valu_factor
        * (
            df_row["SQ_INSTS_VALU_ADD_F64"]
            + df_row["SQ_INSTS_VALU_MUL_F64"]
            + 2 * df_row["SQ_INSTS_VALU_FMA_F64"]
            + df_row["SQ_INSTS_VALU_TRANS_F64"]
        )
    )

    # This line is currently commented out since we do not track INT32/INT64 at this time.
    # Could be tracked in the future.
    ##valu_iops = VALU_FACTOR * (df_row['SQ_INSTS_VALU_INT32'] + df_row['SQ_INSTS_VALU_INT64'])
    ##total_valu_ops = valu_fops + valu_iops

    mfma_fops = mfma_factor * (
        df_row["SQ_INSTS_VALU_MFMA_MOPS_F16"]
        + df_row["SQ_INSTS_VALU_MFMA_MOPS_BF16"]
        + df_row["SQ_INSTS_VALU_MFMA_MOPS_F32"]
        + df_row["SQ_INSTS_VALU_MFMA_MOPS_F64"]
    )

    return (valu_fops, mfma_fops)


# The following lines are used to calculate key LDS, cache, and HBM statistics to help with bottleneck classification.
def find_lds_bytes(df_row):
    num_l2_banks = 32
    bytes_per_workitem = 4

    return (
        num_l2_banks
        * bytes_per_workitem
        * (df_row["SQ_LDS_IDX_ACTIVE"] - df_row["SQ_LDS_BANK_CONFLICT"])
    )


def find_l1d_bytes(df_row):
    bytes_per_cache_line = 64

    return bytes_per_cache_line * df_row["TCP_TOTAL_CACHE_ACCESSES_sum"]


def find_l2_bytes(df_row):
    bytes_per_cache_line = 64

    return bytes_per_cache_line * (
        df_row["TCP_TCC_WRITE_REQ_sum"]
        + df_row["TCP_TCC_READ_REQ_sum"]
        + df_row["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"]
        + df_row["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"]
    )


# The following statistics are commented out, but may prove useful in future versions of the script.
# def find_hbm_stalls(df_row):
#    return df_row[HBM_READ_STALL_COL] + df_row[HBM_WRITE_STALL_COL]

# def find_hbm_requests(df_row):
#    return df_row[HBM_READ_COL] + df_row[HBM_WRITE_COL]

# def find_hbm_total(df_row):
#    return df_row[HBM_READ_STALL_COL] + df_row[HBM_WRITE_STALL_COL] + df_row[HBM_READ_COL] + df_row[HBM_WRITE_COL]

# def find_valu_instructions(df_row):
#    return df_row[VALU_INSTS_COL]

# def find_mfma_instructions(df_row):
#    return df_row[MFMA_INSTS_COL]


def find_hbm_bytes(df_row):
    # bytes of HBM traffic is a bit confusing to parse at first glance
    # but traffic is split between 32B and 64B traffic. reads keep track of 32B reads
    # and all reads, and writes keep track of 64B writes and all writes
    return (
        (df_row["TCC_EA_RDREQ_32B_sum"] * 32)
        + ((df_row["TCC_EA_RDREQ_sum"] - df_row["TCC_EA_RDREQ_32B_sum"]) * 64)
        + (df_row["TCC_EA_WRREQ_64B_sum"] * 64)
        + ((df_row["TCC_EA_WRREQ_sum"] - df_row["TCC_EA_WRREQ_64B_sum"]) * 32)
    )


def parse_top_process_functions_time(tp, top_process_track_id=1):
    """
    this function is used for calculating the total trace time from Omnitrace
    this function will add all the runtimes of slices that are called by the main process
    (usually track_id = 1) and are at function depth = 0. This way, we only consider
    the time where a function from the process we are tracing is running.
    ex: will avoid any kernel/driver initiation time which will not show up as a slice
    in the perfetto ui/traces
    """
    top_process_function_query = (
        f"SELECT * FROM slice WHERE track_id = {top_process_track_id} AND depth = 0"
    )
    function_iter = tp.query(top_process_function_query)
    total_runtime = 0

    for row in function_iter:
        total_runtime += int(row.dur)

    return total_runtime


def parse_merge_multi_gpu_slices(tp, gpu_tracks):
    """
    If multiple GPU kernels are ran back to back, merge the time periods into a single interval
    this is used in bucket parsing to determine if the GPU is busy or not when a kernel is launched from the CPU
    used for tracking cold invoke GPU kernel launches
    gpu_tracks is a map of gpu_id and the corresponding perfetto tracks where those GPU's kernels are run
    """
    gpu_slices_arr = []
    for gpu_track_id in gpu_tracks:
        gpu_slices_query = f"SELECT * FROM slice WHERE track_id = {gpu_track_id}"
        gpu_slice_iter = tp.query(gpu_slices_query)

        for row in gpu_slice_iter:
            gpu_slices_arr.append([row.ts, row.ts + row.dur])

    gpu_slices_arr.sort()
    console_debug("Bottleneck Classifier", f"The GPU slice is: {gpu_slices_arr}")

    merged_gpu_slices_arr = []
    for slice in gpu_slices_arr:
        if not merged_gpu_slices_arr or merged_gpu_slices_arr[-1][1] < slice[0]:
            merged_gpu_slices_arr.append(slice)
        else:
            merged_gpu_slices_arr[-1][1] = max(merged_gpu_slices_arr[-1][1], slice[1])

    return merged_gpu_slices_arr


def calc_bottleneck(
    valu_num_ops,
    mfma_num_ops,
    valu_peak_opsec,
    mfma_peak_opsec,
    num_mem_bytes,
    mem_peak_bw,
    kernel_dur,
):
    """
    This function will calculate the Arithmetic Intensity (AI) and flops/sec of the kernel
    and determine where it is bottlenecked (bandwidth or compute), the percent of peak of bandwidth,
    and current arithmetic intensity for this kernel
    """
    # corner case: if the kernel doesn't use any memory ops, we define the AI and FLOPs to be 0
    # this happens when the LDS is not used, and you're calculating AI relative to LDS
    if num_mem_bytes == 0:
        return "no_mem", 0, 0

    knee_ai = -1
    curr_ai = -1
    compute_str = ""
    kernel_opsec = -1
    attainable_kernel_opsec = -1

    total_flops = valu_num_ops + mfma_num_ops

    # classify each kernel by if it's VALU or MFMA compute dominated
    # the knee AI is point in the roofline where the bandwidth bound (diagnol line) transitions to
    # compute bound flat roof (horizontal line).
    if valu_num_ops > mfma_num_ops:
        knee_ai = valu_peak_opsec / mem_peak_bw
        curr_ai = total_flops / num_mem_bytes
        kernel_opsec = total_flops / kernel_dur
        attainable_kernel_opsec = calc_attainable_opsec(
            curr_ai, valu_peak_opsec, mem_peak_bw
        )
        compute_str = "valu_compute"
    else:
        knee_ai = mfma_peak_opsec / mem_peak_bw
        curr_ai = total_flops / num_mem_bytes
        kernel_opsec = total_flops / kernel_dur
        attainable_kernel_opsec = calc_attainable_opsec(
            curr_ai, mfma_peak_opsec, mem_peak_bw
        )
        compute_str = "mfma_compute"

    # Calculate the percentage of peak ops per second. In other words, how close to the attainable peak is this kernel?
    kernel_opsec_pop = (
        0 if attainable_kernel_opsec == 0 else kernel_opsec / attainable_kernel_opsec
    )

    # Now to figure out if you are actually bandwidth bound.
    # if you're less than the knee, you're bandwidth bound and return that, not the compute_str value.
    if curr_ai < knee_ai:
        return "bandwidth", kernel_opsec_pop, curr_ai
    else:
        return compute_str, kernel_opsec_pop, curr_ai


def open_proto_file(filename):
    tp = TraceProcessor(trace=(filename))
    return tp


def calc_attainable_opsec(op_intensity, peak_opsec, peak_bw):
    return np.minimum(peak_opsec, peak_bw * op_intensity)


def parse_gpu_track_map(tp):
    """
    Create a map where the keys are the gpu device id, and values are the tracks that
    belong to that device
    """
    gpu_track_query = "SELECT * FROM track WHERE name LIKE '%HIP%'"
    gpu_track_iter = tp.query(gpu_track_query)
    gpu_track_map = {}

    # gpu tracks are something like 'HIP Activity Device <#>, Queue <#>'
    # we only care about device number
    for row in gpu_track_iter:
        name_numbers = list(map(int, re.findall(r"\d+", row.name)))
        gpu_id = name_numbers[0]
        gpu_track_map.setdefault(gpu_id, []).append(row.id)
    return gpu_track_map


def calc_delay(caller_ts, caller_dur, callee_ts):
    """
    Used for calculating the delay between the end of CPU's function call to invoke a kernel
    and the start of the kernel actually beginning to run on the GPU
    caller is the CPU, callee is the GPU
    ex: end of hipMemcpyWithStream (CPU function) and start of copyDeviceToHost (GPU kernel)
    end of hipLaunchKernel (CPU function) and start of Cijk_Ailk_Bjlk... (GPU kernel)
    """
    nominal_delay = callee_ts - (caller_ts + caller_dur)
    return nominal_delay if nominal_delay > 0 else 0


def get_bottleneck_classification(omniperf_dir, omnitrace_dir):
    treshold_ratio = 0.8
    bc_obj = Bottleneck_Classification(omniperf_dir, omnitrace_dir, treshold_ratio)
    # make sure that the gpu ids match between omnitrace and omniperf
    assert (
        list(bc_obj.omnitrace_data["gpu_ids"]).sort()
        == list(bc_obj["gpu_bounds_time"].keys()).sort()
    )

#!/usr/bin/env python3

# Author: August Ning (auguning@amd.com)
# Modified by: Daniel Chang (daniel.chang@amd.com)
# Last Updated: 01/29/24
# Summer 2023 Co-op project for RIKEN Fugaku-Next
# This script will parse omnitrace proto files and omniperf databases, combine their data
# to provide detailed runtime classification for CPU-GPU systems
# and GPU per kernel bottleneck and performance analysis
#
# Since August's initial commit, I (Daniel) have made major modifications to the bottleneck classification methodology.
# Comments on those changes will be explained explicitly. Lastly, the code was tested on MI200's and HAS NOT been
# validated for MI300.

import os
import re
from perfetto.trace_processor import TraceProcessor
import pandas as pd
import numpy as np
import csv 

################################ omnitrace ################################
################################ omnitrace ################################
################################ omnitrace ################################
# This section of code is used for Omnitrace. Specifically helper functions and the actual parsing of Omnitrace data.

# will search for this folder for .proto files
BASE_TRACE_FOLDER='omnitrace'

def open_proto_file(filename):
    tp = TraceProcessor(trace=(filename))
    return tp

# create a map where the keys are the gpu device id, and values are the tracks that 
# belong to that device 
def parse_gpu_track_map(tp):
    gpu_track_query = "SELECT * FROM track WHERE name LIKE '%HIP%'"
    gpu_track_iter = tp.query(gpu_track_query)
    gpu_track_map = {}

    # gpu tracks are something like 'HIP Activity Device <#>, Queue <#>'
    # we only care about device number
    for row in gpu_track_iter:
        name_numbers = list(map(int, re.findall(r'\d+', row.name)))
        gpu_id = name_numbers[0]
        gpu_track_map.setdefault(gpu_id, []).append(row.id)
    return gpu_track_map

# if multiple GPU kernels are ran back to back, merge the time periods into a single interval
# this is used in bucket parsing to determine if the GPU is busy or not when a kernel is launched from the CPU
# used for tracking cold invoke GPU kernel launches
# gpu_tracks is a map of gpu_id and the corresponding perfetto tracks where those GPU's kernels are run
def parse_merge_multi_gpu_slices(tp, gpu_tracks):
    gpu_slices_arr = []
    for gpu_track_id in gpu_tracks:
        gpu_slices_query = f"SELECT * FROM slice WHERE track_id = {gpu_track_id}"
        gpu_slice_iter = tp.query(gpu_slices_query)

        for row in gpu_slice_iter:
            gpu_slices_arr.append([row.ts, row.ts+row.dur])

    gpu_slices_arr.sort()
    #print('The GPU slice is: ', gpu_slices_arr)

    # merge any contiguous intervals. This code I copied from leetcode
    merged_gpu_slices_arr = []
    for slice in gpu_slices_arr:
        if not merged_gpu_slices_arr or merged_gpu_slices_arr[-1][1] < slice[0]:
            merged_gpu_slices_arr.append(slice)
        else:
            merged_gpu_slices_arr[-1][1] = max(merged_gpu_slices_arr[-1][1], slice[1])

    return merged_gpu_slices_arr

# this function is used for calculating the total trace time from Omnitrace
# this function will add all the runtimes of slices that are called by the main process
# (usually track_id = 1) and are at function depth = 0. This way, we only consider
# the time where a function from the process we are tracing is running.
# ex: will avoid any kernel/driver initiation time which will not show up as a slice
# in the perfetto ui/traces
def parse_top_process_functions_time(tp, top_process_track_id=1):
    top_process_function_query = f"SELECT * FROM slice WHERE track_id = {top_process_track_id} AND depth = 0"
    function_iter = tp.query(top_process_function_query)
    total_runtime = 0

    for row in function_iter:
        total_runtime += int(row.dur)

    return total_runtime

# used for calculating the delay between the end of CPU's function call to invoke a kernel
# and the start of the kernel actually beginning to run on the GPU
# caller is the CPU, callee is the GPU
# ex: end of hipMemcpyWithStream (CPU function) and start of copyDeviceToHost (GPU kernel)
# end of hipLaunchKernel (CPU function) and start of Cijk_Ailk_Bjlk... (GPU kernel)
def calc_delay(caller_ts, caller_dur, callee_ts):
    nominal_delay = callee_ts - (caller_ts + caller_dur)
    return nominal_delay if nominal_delay > 0 else 0

# this is a workaround function for tracking GPU time on omnitrace due to errors
# where all the GPU kernels aren't collected by omnitrace
# there is a track that keeps track of a GPU busy counter, and we use this as a proxy
# for GPU runtime
def parse_gpu_busy(tp, gpu_ids):
    gpu_names_query = "SELECT * FROM track WHERE name LIKE 'GPU Busy%'"
    gpu_busy_query = "SELECT * FROM track JOIN counter ON track.id = counter.track_id WHERE name LIKE 'GPU Busy%' ORDER BY counter.ts"
    gpu_names_iter = tp.query(gpu_names_query)
    gpu_busy_iter = tp.query(gpu_busy_query)
    gpu_names_track_ids = []
    gpu_busy_tuple_map = {}
    gpu_busy_dur_map = {}

    for row in gpu_names_iter:
        gpu_names_track_ids.append(row.id)

    # keys are 9, 13; vals are 2, 3
    gpu_ids_map = dict(zip(gpu_names_track_ids, gpu_ids))

    # collect a tuple of the time stamp and the gpu busy value
    for row in gpu_busy_iter:
        gpu_busy_tuple_map.setdefault(gpu_ids_map[row.track_id], []).append((row.ts, row.value))

    for gpu_id, busy_tuples in gpu_busy_tuple_map.items():
        gpu_busy_dur_map[gpu_id] = 0

        for i in range(len(busy_tuples) - 1):
            if busy_tuples[i][1] > 0:
                gpu_busy_dur_map[gpu_id] += (busy_tuples[i + 1][0] - busy_tuples[i][0])

    # the returned map is the gpu_id and the cumulative time where the GPU busy signal was greater than 0
    return gpu_busy_dur_map

# buckets for omnitrace time analysis
# cpu-gpu-mem: time where spent where cpu and gpu are copy/setting memory between each other
# this is further broken down by the direction (cpu to gpu, gpu to cpu) and gpu-gpu communication
# gpu-kernel: time gpu is spent running work that is invoked by the cpu by `hipLaunchKernel`
# kernel-invoke-overhead: gpu kernel invocation overhead (time after `hipLaunchKernel` ends 
#                          on the CPU and the corresponding GPU kernel starts running
# cpu: any time other time that traced by omnitrace
def parse_omnitrace(TRACE_FILE_NAME):
    tp = open_proto_file(os.path.join(os.getcwd(), BASE_TRACE_FOLDER, TRACE_FILE_NAME))

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

    print('The trace stats iter is: ', trace_stats_iter)
    print('The gpu tracks map is: ', gpu_tracks_map)

    # for multi gpu, classify the rows/slices by which GPU they are invoked on
    # GPU kernels for a single GPU may show up on multiple GPU tracks
    gpu_slices_map = {}
    for row in trace_stats_iter:
        for gpu_id in gpu_tracks_map.keys():
            if row.callee_track_id in gpu_tracks_map[gpu_id]:
                gpu_slices_map.setdefault(gpu_id, []).append(row)
                #print('I am appending with the row: ', row)

    # we use the CPU's function name to determine the type of workload being invoked on the GPU
    # this is since the gpu kernel's names are intractable to parse
    # these are common substrings in the Proto file for kernel and memory communication
    launch_kernel_substring = ['launchkernel']
    cpu_gpu_mem_substrings = ['memcpy', 'memset', 'malloc']
    
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
        merged_kernel_slices_map[gpu_id] = parse_merge_multi_gpu_slices(tp, gpu_tracks)

    for gpu_id in gpu_tracks_map.keys():
        for row in gpu_slices_map[gpu_id]:
            curr_gpu_kernel_i = 0
            # Look for any of the substrings above so we can accumulate those run times.
            if any(substring in row.caller_name.lower() for substring in cpu_gpu_mem_substrings):
                # for memory ops, the gpu memory kernel may start before the cpu's function ends (there's overlap)
                # avoid double counting the overlap
                mem_op_time = (row.callee_ts + row.callee_dur) - row.caller_ts
                if (row.callee_name.lower() == 'copydevicetodevice'):
                    gpu_gpu_comm_map.setdefault(gpu_id, []).append(row.callee_dur)
                elif (row.callee_name.lower() == 'copydevicetohost'):
                    device_to_host_mem_map.setdefault(gpu_id, []).append(mem_op_time)
                elif (row.callee_name.lower() == 'copyhosttodevice' or row.caller_name.lower() == 'hipmemset'):
                    host_to_device_mem_map.setdefault(gpu_id, []).append(mem_op_time)

                # fillbuffer calls are controversial since some are inherent in the 
                # program, but are also often inserted by ROCM profiler (and these calls would
                # not show up when running without omnitrace/omniperf). We ignore them since
                # they do not significantly affect runtime
                elif (row.callee_name.lower() == 'fillbuffer'):
                    continue
                else:
                    raise Exception('not classified')
                
            # case when kernel is launched by the cpu via hiplaunchkernel
            elif any(substring in row.caller_name.lower() for substring in launch_kernel_substring):
                # r/nccl calls represent GPU to GPU communications but are launched by the CPU
                if ('nccl' in row.callee_name.lower()) or ('rccl' in row.callee_name.lower()):
                    gpu_gpu_comm_map.setdefault(gpu_id, []).append(row.callee_dur)
                else:
                    # all other calls we classify as compute
                    gpu_kernel_map.setdefault(gpu_id, []).append(row.callee_dur)

                invoke_time = calc_delay(row.caller_ts, row.caller_dur, row.callee_ts)
                
                # find the relevant slice where the gpu kernel's end time is greater than 
                # the cpu's launchkernel's slice end time
                while merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][1] < (row.caller_ts + row.caller_dur):
                    curr_gpu_kernel_i += 1
                                
                # if the launchkernel call ended while a gpu slice is running, this is a stalled invoke
                # otherwise this is a cold invoke
                if merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][0] <= (row.caller_ts + row.caller_dur) \
                    and (row.caller_ts + row.caller_dur) <= merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][1]:
                    gpu_kernel_stall_invoke_delay_map.setdefault(gpu_id, []).append(invoke_time)
                else:
                    # nothing is running on the GPU, so this is a cold invoke
                    # sanity check: for this case the happen, the launchkernel end time must happen
                    # before the next kernel's start time
                    assert(row.caller_ts + row.caller_dur <= merged_kernel_slices_map[gpu_id][curr_gpu_kernel_i][0])
                    gpu_kernel_cold_invoke_overhead_map.setdefault(gpu_id, []).append(invoke_time)
        
        ## if your omnitrace database has bugs and your GPU time is being undercounted,
        ## you can choose to return gpu_busy_time_map istead of gpu_kernel_map
        # gpu_busy_time_map = parse_gpu_busy(tp, gpu_tracks_map.keys())

    total_traced_runtime = parse_top_process_functions_time(tp)

    return (gpu_tracks_map.keys(), total_traced_runtime, gpu_kernel_map, host_to_device_mem_map, \
            device_to_host_mem_map, gpu_gpu_comm_map, gpu_kernel_cold_invoke_overhead_map)

################################ omniperf ################################
################################ omniperf ################################
################################ omniperf ################################
# This section of code is used for Omniperf. It specifically, looks as the pmc_perf.csv file that Omniperf generates
# to figure out Arithmetic Intensities (AI) and also the performance counters are used for identifying bottlenecks.

# The OMNIPERF_DB_BASE is the primary folder all omniperf folders should be placed in.
OMNIPERF_DB_BASE='omniperf'
PMC_PERF_FILE='pmc_perf.csv'
ROOFLINE_DATA_FILE = 'roofline.csv'


# constants to help index columns in pmc_perf.csv
NAME_COL = 'KernelName'
GPU_ID_COL = 'gpu-id'

LDS_HITS_COL = 'SQ_LDS_IDX_ACTIVE'
LDS_MISS_COL = 'SQ_LDS_BANK_CONFLICT'
L1D_COL = 'TCP_TOTAL_CACHE_ACCESSES_sum'
L2_WRITE_COL = 'TCP_TCC_WRITE_REQ_sum'
L2_ATOMIC_WRET_REQ_COL = 'TCP_TCC_ATOMIC_WITH_RET_REQ_sum'
L2_ATOMIC_WORET_REQ_COL = 'TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum'
L2_READ_COL = 'TCP_TCC_READ_REQ_sum'

# DANIEL: ADDING NEW VARIABLES FROM OMNIPERF
HBM_READ_STALL_COL = 'TCC_EA_RDREQ_DRAM_CREDIT_STALL_sum'
HBM_WRITE_STALL_COL = 'TCC_EA_WRREQ_DRAM_CREDIT_STALL_sum'
HBM_READ_COL = 'TCC_EA_RDREQ_DRAM_sum'
HBM_WRITE_COL = 'TCC_EA_WRREQ_DRAM_sum'
VALU_INSTS_COL = 'SQ_INSTS_VALU'
MFMA_INSTS_COL = 'SQ_INSTS_MFMA'

BYTES_PER_CACHE_LINE = 64 # this is what I assume 64 comes from
NUM_L2_BANKS = 32
LDS_MAGIC_NUMBER = 4 # no idea why there is a 4 here

VALU_FACTOR = 64
MFMA_FACTOR = 512

# the following functions calculate the FLOPs and memory traffic
# according to hierarchical roofline modeling definitions 
# used in the super computing 2022 tutorial and omnitrace
# IMPORTANT NOTE: These calculations are for MI200 GPUs

def find_ops_bytes_mi200(df_row):
    valu_fops = VALU_FACTOR * \
                (df_row["SQ_INSTS_VALU_ADD_F16"] + \
                df_row["SQ_INSTS_VALU_MUL_F16"] + \
                2 * df_row["SQ_INSTS_VALU_FMA_F16"] + \
                df_row["SQ_INSTS_VALU_TRANS_F16"]) + \
                VALU_FACTOR * \
                (df_row["SQ_INSTS_VALU_ADD_F32"] + \
                df_row["SQ_INSTS_VALU_MUL_F32"] + \
                2 * df_row["SQ_INSTS_VALU_FMA_F32"] + \
                df_row["SQ_INSTS_VALU_TRANS_F32"]) + \
                VALU_FACTOR * \
                (df_row["SQ_INSTS_VALU_ADD_F64"] + \
                df_row["SQ_INSTS_VALU_MUL_F64"] + \
                2 * df_row["SQ_INSTS_VALU_FMA_F64"] + \
                df_row["SQ_INSTS_VALU_TRANS_F64"])

    # This line is currently commented out since we do not track INT32/INT64 at this time.
    # Could be tracked in the future.
    ##valu_iops = VALU_FACTOR * (df_row['SQ_INSTS_VALU_INT32'] + df_row['SQ_INSTS_VALU_INT64'])
    ##total_valu_ops = valu_fops + valu_iops

    mfma_fops = MFMA_FACTOR * (df_row["SQ_INSTS_VALU_MFMA_MOPS_F16"] + df_row["SQ_INSTS_VALU_MFMA_MOPS_BF16"] + \
                               df_row["SQ_INSTS_VALU_MFMA_MOPS_F32"] + df_row["SQ_INSTS_VALU_MFMA_MOPS_F64"])

    return (valu_fops, mfma_fops)

# The following lines are used to calculate key LDS, cache, and HBM statistics to help with bottleneck classification.
def find_lds_bytes(df_row):
    return NUM_L2_BANKS * LDS_MAGIC_NUMBER * \
        (df_row[LDS_HITS_COL] - df_row[LDS_MISS_COL])

def find_l1d_bytes(df_row):
    return BYTES_PER_CACHE_LINE * df_row[L1D_COL]

def find_l2_bytes(df_row):
    return BYTES_PER_CACHE_LINE * \
            (df_row[L2_WRITE_COL] + df_row[L2_WRITE_COL] + \
              df_row[L2_ATOMIC_WRET_REQ_COL] + df_row[L2_ATOMIC_WORET_REQ_COL] )

def find_hbm_bytes(df_row):
    # bytes of HBM traffic is a bit confusing to parse at first glance
    # but traffic is split between 32B and 64B traffic. reads keep track of 32B reads
    # and all reads, and writes keep track of 64B writes and all writes
    return  (
            (df_row["TCC_EA_RDREQ_32B_sum"] * 32)
            + ((df_row["TCC_EA_RDREQ_sum"] - df_row["TCC_EA_RDREQ_32B_sum"]) * 64)
            + (df_row["TCC_EA_WRREQ_64B_sum"] * 64)
            + ((df_row["TCC_EA_WRREQ_sum"] - df_row["TCC_EA_WRREQ_64B_sum"]) * 32)
            )

# The following statistics are commented out, but may prove useful in future versions of the script.
#def find_hbm_stalls(df_row):
#    return df_row[HBM_READ_STALL_COL] + df_row[HBM_WRITE_STALL_COL]

#def find_hbm_requests(df_row):
#    return df_row[HBM_READ_COL] + df_row[HBM_WRITE_COL]

#def find_hbm_total(df_row):
#    return df_row[HBM_READ_STALL_COL] + df_row[HBM_WRITE_STALL_COL] + df_row[HBM_READ_COL] + df_row[HBM_WRITE_COL]

#def find_valu_instructions(df_row):
#    return df_row[VALU_INSTS_COL]

#def find_mfma_instructions(df_row):
#    return df_row[MFMA_INSTS_COL]

def calc_attainable_opsec(op_intensity, peak_opsec, peak_bw):
    return np.minimum(peak_opsec, peak_bw * op_intensity)

# this function will calculate the Arithmetic Intensity (AI) and flops/sec of the kernel
# and determine where it is bottlenecked (bandwidth or compute), the percent of peak of bandwidth,
# and current arithmetic intensity for this kernel
def calc_bottleneck(valu_num_ops, mfma_num_ops, valu_peak_opsec, mfma_peak_opsec, num_mem_bytes, mem_peak_bw, kernel_dur):
    # corner case: if the kernel doesn't use any memory ops, we define the AI and FLOPs to be 0
    # this happens when the LDS is not used, and you're calculating AI relative to LDS
    if num_mem_bytes == 0:
        return 'no_mem', 0, 0
    
    knee_ai = -1
    curr_ai = -1
    compute_str = ''
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
        attainable_kernel_opsec = calc_attainable_opsec(curr_ai, valu_peak_opsec, mem_peak_bw)
        compute_str = 'valu_compute'
    else:
        knee_ai = mfma_peak_opsec / mem_peak_bw
        curr_ai = total_flops / num_mem_bytes
        kernel_opsec = total_flops / kernel_dur
        attainable_kernel_opsec = calc_attainable_opsec(curr_ai, mfma_peak_opsec, mem_peak_bw)
        compute_str = 'mfma_compute'

    # Calculate the percentage of peak ops per second. In other words, how close to the attainable peak is this kernel?
    kernel_opsec_pop = 0 if attainable_kernel_opsec == 0 else kernel_opsec / attainable_kernel_opsec

    # Now to figure out if you are actually bandwidth bound.
    # if you're less than the knee, you're bandwidth bound and return that, not the compute_str value.
    if curr_ai < knee_ai:
        return 'bandwidth', kernel_opsec_pop, curr_ai
    else:
        return compute_str, kernel_opsec_pop, curr_ai
    
# return a map of gpu_ids and corresponding roofline maxes
def parse_roofline_data(roof_df, gpu_ids):
    roofline_data_map = {}

    # the device id in roofline.csv (0, 1) does not match the gpu ids in pmc_perf.csv (2, 3)
    # so use two separate counters
    roofline_counter = 0
    for id in gpu_ids:
        peak_bw_lds = roof_df['LDSBw'][roofline_counter]
        peak_bw_l1d = roof_df['L1Bw'][roofline_counter]
        peak_bw_l2 = roof_df['L2Bw'][roofline_counter]
        peak_bw_hbm = roof_df['HBMBw'][roofline_counter]
        # These two values are the peak valu and mfma performance for a certain data type.
        # They need to be changed if you are using a different data type.
        # For VALU, The options are: FP64Flops and FP32Flops
        # For MFMA, the options are: MFMAF64Flops, MFMAF32Flops, MFMAF16Flops, MFMABF16Flops
        peak_valu_flops = roof_df['FP32Flops'][roofline_counter]
        peak_mfma_flops = roof_df['MFMAF32Flops'][roofline_counter]

        roofline_data_map[id] = (peak_bw_lds, peak_bw_l1d, peak_bw_l2, peak_bw_hbm, peak_valu_flops, peak_mfma_flops)

        roofline_counter += 1
    
    return roofline_data_map

# The utilization ratio is used to set a threshold for percentage of the empirical peak FLOPS/s.
# Used to classify a kernel as performing well and underperforming.
def parse_omniperf(OMNIPERF_DB_FOLDER_NAME, util_ratio=0.8):
    perf_df = pd.read_csv(os.path.join(OMNIPERF_DB_BASE, OMNIPERF_DB_FOLDER_NAME, PMC_PERF_FILE))
    roofline_df = pd.read_csv(os.path.join(OMNIPERF_DB_BASE, OMNIPERF_DB_FOLDER_NAME, ROOFLINE_DATA_FILE))

    START_TS_COL = 'BeginNs'
    END_TS_COL = 'EndNs'

    # go row by row in pmc_perf to parse the relevant performance counters per kernel to
    # collect the parameters needed for hierarchical rooflines by calling the helper functions
    # kernel data map uses gpu-id as the key, and maps to a list of data tuples
    kernel_data_map = {}
    for i, row in perf_df.iterrows():
        gpu_id = row['gpu-id']
        curr_kernel_name = row[NAME_COL]
        curr_kernel_dur = (row[END_TS_COL] - row[START_TS_COL])

        valu_num_ops, mfma_num_ops = find_ops_bytes_mi200(row)
        data_row = (curr_kernel_name, curr_kernel_dur, valu_num_ops, mfma_num_ops, find_lds_bytes(row), \
                        find_l1d_bytes(row), find_l2_bytes(row), find_hbm_bytes(row))

        kernel_data_map.setdefault(gpu_id, []).append(data_row)

    roofline_max_map = parse_roofline_data(roofline_df, kernel_data_map.keys())

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

        peak_bw_lds, peak_bw_l1d, peak_bw_l2, peak_bw_hbm, peak_valu_flops, peak_mfma_flops = roofline_max_map[gpu_id]
        gpu_gpu_comm_time_map[gpu_id] = 0

        for tups in data_tups:
            kernel_name, kernel_dur, valu_num_ops, mfma_num_ops, num_lds_bytes, num_l1d_bytes, num_l2_bytes, num_hbm_bytes = tups

            # Similar to the code for Omnitrace earlier, ignore fillbuffer calls
            if ('fillbuffer' in kernel_name.lower()):
                continue
            # we don't classify rccl calls in the gpu to prevent
            # double counting gpu-gpu communication time
            elif ('rccl' in kernel_name.lower() or 'nccl' in kernel_name.lower()):
                gpu_gpu_comm_time_map[gpu_id] += kernel_dur
                continue

            # Each of these lines is used to help calculate a kernel's bottleneck  (bandwidth or compute) at the 4
            # memory levels (LDS, L1, L2, and HBM). These values will be used to classify the bottlenecks later
            # in the code.
            hbm_bound, hbm_pop, hbm_ai = calc_bottleneck(valu_num_ops, mfma_num_ops, peak_valu_flops,
                                                             peak_mfma_flops, num_hbm_bytes, peak_bw_hbm, kernel_dur)
            l2_bound, l2_pop, l2_ai = calc_bottleneck(valu_num_ops, mfma_num_ops, peak_valu_flops,
                                                          peak_mfma_flops, num_l2_bytes, peak_bw_l2, kernel_dur)
            l1d_bound, l1d_pop, l1d_ai = calc_bottleneck(valu_num_ops, mfma_num_ops, peak_valu_flops,
                                                             peak_mfma_flops, num_l1d_bytes, peak_bw_l1d, kernel_dur)
            lds_bound, lds_pop, lds_ai = calc_bottleneck(valu_num_ops, mfma_num_ops, peak_valu_flops,
                                                             peak_mfma_flops, num_lds_bytes, peak_bw_lds, kernel_dur)

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
            if (num_hbm_bytes > mfma_num_ops and num_hbm_bytes > valu_num_ops):
                # Used to debug
                #print('I am in the HBM conditional', num_hbm_bytes, mfma_num_ops, valu_num_ops)
                if (hbm_bound == 'bandwidth'):
                    if (hbm_ai == 0):
                        no_flops_time[3] += kernel_dur
                    # How close was this kernel to the peak and is that below the set utilization ratio?
                    elif (hbm_pop < util_ratio):
                        below_threshold_time[3] += kernel_dur
                    else:
                        above_threshold_time[3] += kernel_dur
                elif (l2_bound == 'bandwidth'):
                    if (l2_ai == 0):
                        no_flops_time[2] += kernel_dur
                    elif (l2_pop < util_ratio):
                        below_threshold_time[2] += kernel_dur
                    else:
                        above_threshold_time[2] += kernel_dur
                elif (l1d_bound == 'bandwidth'):
                    if (l1d_ai == 0):
                        no_flops_time[1] += kernel_dur
                    elif (l1d_pop < util_ratio):
                        below_threshold_time[1] += kernel_dur
                    else:
                        above_threshold_time[1] += kernel_dur
                elif (lds_bound == 'bandwidth'):
                    if (lds_ai == 0):
                        no_flops_time[0] += kernel_dur
                    elif (lds_pop < util_ratio):
                        below_threshold_time[0] += kernel_dur
                    else:
                        above_threshold_time[0] += kernel_dur
                elif (valu_num_ops > mfma_num_ops):
                    if (hbm_pop < util_ratio):
                        below_threshold_time[4] += kernel_dur
                    else:
                        above_threshold_time[4] += kernel_dur
                else:
                    if (hbm_pop < util_ratio):
                        below_threshold_time[5] += kernel_dur
                    else:
                        above_threshold_time[5] += kernel_dur
            elif (mfma_num_ops > num_hbm_bytes and mfma_num_ops > valu_num_ops):
                # Used to debug
                #print('I am in the MFMA conditional', num_hbm_bytes, mfma_num_ops, valu_num_ops)
                if (mfma_num_ops > valu_num_ops):
                    if (hbm_pop < util_ratio):
                        below_threshold_time[5] += kernel_dur
                    else:
                        above_threshold_time[5] += kernel_dur
                elif (hbm_bound == 'bandwidth'):
                    if (hbm_ai == 0):
                        no_flops_time[3] += kernel_dur
                    elif (hbm_pop < util_ratio):
                        below_threshold_time[3] += kernel_dur
                    else:
                        above_threshold_time[3] += kernel_dur
                elif (l2_bound == 'bandwidth'):
                    if (l2_ai == 0):
                        no_flops_time[2] += kernel_dur
                    elif (l2_pop < util_ratio):
                        below_threshold_time[2] += kernel_dur
                    else:
                        above_threshold_time[2] += kernel_dur
                elif (l1d_bound == 'bandwidth'):
                    if (l1d_ai == 0):
                        no_flops_time[1] += kernel_dur
                    elif (l1d_pop < util_ratio):
                        below_threshold_time[1] += kernel_dur
                    else:
                        above_threshold_time[1] += kernel_dur
                elif (lds_bound == 'bandwidth'):
                    if (lds_ai == 0):
                        no_flops_time[0] += kernel_dur
                    elif (lds_pop < util_ratio):
                        below_threshold_time[0] += kernel_dur
                    else:
                        above_threshold_time[0] += kernel_dur
                else:
                    if (hbm_pop < util_ratio):
                        below_threshold_time[4] += kernel_dur
                    else:
                        above_threshold_time[4] += kernel_dur
            elif (valu_num_ops > num_hbm_bytes and valu_num_ops > mfma_num_ops):
                # Used to debug
                #print('I am in the VALU conditional', num_hbm_bytes, mfma_num_ops, valu_num_ops)
                if (valu_num_ops > mfma_num_ops):
                    if (hbm_pop < util_ratio):
                        below_threshold_time[4] += kernel_dur
                    else:
                        above_threshold_time[4] += kernel_dur
                elif (hbm_bound == 'bandwidth'):
                    if (hbm_ai == 0):
                        no_flops_time[3] += kernel_dur
                    elif (hbm_pop < util_ratio):
                        below_threshold_time[3] += kernel_dur
                    else:
                        above_threshold_time[3] += kernel_dur
                elif (l2_bound == 'bandwidth'):
                    if (l2_ai == 0):
                        no_flops_time[2] += kernel_dur
                    elif (l2_pop < util_ratio):
                        below_threshold_time[2] += kernel_dur
                    else:
                        above_threshold_time[2] += kernel_dur
                elif (l1d_bound == 'bandwidth'):
                    if (l1d_ai == 0):
                        no_flops_time[1] += kernel_dur
                    elif (l1d_pop < util_ratio):
                        below_threshold_time[1] += kernel_dur
                    else:
                        above_threshold_time[1] += kernel_dur
                elif (lds_bound == 'bandwidth'):
                    if (lds_ai == 0):
                        no_flops_time[0] += kernel_dur
                    elif (lds_pop < util_ratio):
                        below_threshold_time[0] += kernel_dur
                    else:
                        above_threshold_time[0] += kernel_dur
                else:
                    if (hbm_pop < util_ratio):
                        below_threshold_time[5] += kernel_dur
                    else:
                        above_threshold_time[5] += kernel_dur
            else:
                if (hbm_bound == 'bandwidth'):
                    if (hbm_ai == 0):
                        no_flops_time[3] += kernel_dur
                    elif (hbm_pop < util_ratio):
                        below_threshold_time[3] += kernel_dur
                    else:
                        above_threshold_time[3] += kernel_dur
                elif (l2_bound == 'bandwidth'):
                    if (l2_ai == 0):
                        no_flops_time[2] += kernel_dur
                    elif (l2_pop < util_ratio):
                        below_threshold_time[2] += kernel_dur
                    else:
                        above_threshold_time[2] += kernel_dur
                elif (l1d_bound == 'bandwidth'):
                    if (l1d_ai == 0):
                        no_flops_time[1] += kernel_dur
                    elif (l1d_pop < util_ratio):
                        below_threshold_time[1] += kernel_dur
                    else:
                        above_threshold_time[1] += kernel_dur
                elif (lds_bound == 'bandwidth'):
                    if (lds_ai == 0):
                        no_flops_time[0] += kernel_dur
                    elif (lds_pop < util_ratio):
                        below_threshold_time[0] += kernel_dur
                    else:
                        above_threshold_time[0] += kernel_dur
                elif (valu_num_ops > mfma_num_ops):
                    if (hbm_pop < util_ratio):
                        below_threshold_time[4] += kernel_dur
                    else:
                        above_threshold_time[4] += kernel_dur
                else:
                    if (hbm_pop < util_ratio):
                        below_threshold_time[5] += kernel_dur
                    else:
                        above_threshold_time[5] += kernel_dur

        bottleneck_classified_time = [no_flops_time, below_threshold_time, above_threshold_time]
        bottleneck_classified_time_map[gpu_id] = bottleneck_classified_time

    return (bottleneck_classified_time_map, gpu_gpu_comm_time_map)


################## beginning of main() ##################
# These four variables need to be changed. The first is the PROTO file generated by Omnitrace, the second
# is the directory containing the omniperf CSV files (which should be in a folder labeled "omniperf"), the third
# is the CSV file name you want, and the last is the percentage of the peak empirical FLOPS/s you want to
# use as the threshold to compare against.
OMNITRACE_TRACE_FILE_NAME = 'MY_PERFETTO_FILE.proto'
OMNIPERF_DB_FOLDER_NAME = 'MY_OMNIPERF_FOLDER'
OUTPUT_FILE_NAME= 'MY_FILENAME.csv'
UTIL_THRESHOLD_RATIO = 0.8

ot_gpu_ids, ot_total_time, ot_gpu_time_map, ot_host_device_time_map, ot_device_time_host_map, \
    ot_gpu_gpu_comm_time_map, ot_gpu_invoke_time_map = parse_omnitrace(OMNITRACE_TRACE_FILE_NAME)
op_gpu_bounds_time_map, op_gpu_gpu_comm_time_map = parse_omniperf(OMNIPERF_DB_FOLDER_NAME, UTIL_THRESHOLD_RATIO)

# make sure that the gpu ids match between omnitrace and omniperf
assert (list(ot_gpu_ids).sort() == list(op_gpu_bounds_time_map.keys()).sort())

# Print out all of the statistics we have calculated of accumulated.
# The else statements are a corner case for when no value was accumulated.
print(f'omnitrace total traced time: {ot_total_time}')
print(f"ot_host_device_time_map: {ot_host_device_time_map}")
print(f'omnitrace device to host time: {(ot_device_time_host_map)}')
print(f'omnitrace gpu-gpu comm time: {(ot_gpu_gpu_comm_time_map)}')
for gpu_id in ot_gpu_ids:
    print(f'gpu_id: {gpu_id}')
    if (ot_gpu_time_map):
        print(f'omnitrace total GPU time: {sum(ot_gpu_time_map[gpu_id])}')
    else:
        print(f'omnitrace total GPU time: 0')
    if (ot_host_device_time_map):
        print(f'omnitrace host to device time: {sum(ot_host_device_time_map[gpu_id])}')
    else:
        print(f'omnitrace host to device time: 0')
    if (ot_device_time_host_map):
        print(f'omnitrace device to host time: {sum(ot_device_time_host_map[gpu_id])}')
    else:
        print(f'omnitrace device to host time: 0')
    if (ot_gpu_gpu_comm_time_map):
        print(f'omnitrace gpu-gpu comm time: {sum(ot_gpu_gpu_comm_time_map[gpu_id])}')
    else:
        print(f'omnitrace gpu-gpu comm time: 0')
    if (ot_gpu_invoke_time_map):
        print(f'omnitrace cold invoke time: {sum(ot_gpu_invoke_time_map[gpu_id])}')
    else:
        print(f'omnitrace cold invoke time: 0')
    print(f'omniperf gpu-gpu comm time: {op_gpu_gpu_comm_time_map[gpu_id]}')
    print()
    op_no_flops_time, op_below_util_time, op_above_util_time = op_gpu_bounds_time_map[gpu_id]
    print(f'omniperf no flops time: {op_no_flops_time}')
    print(f'omniperf under threshold pop time: {op_below_util_time}')
    print(f'omniperf above threshold pop time: {op_above_util_time}')
    print()

output_headers = [
    'gpu_id', 'ot_total_trace_time', 'ot_gpu_time', 'ot_host_device_time', 'ot_device_host_time', 
    'ot_gpu_gpu_comm_time', 'ot_invoke_time', 'op_gpu_time', 'op_gpu_gpu_comm_time', 'ot_util_threshold', 
    'ot_no_flops_lds_time', 'ot_no_flops_gl1_time', 'ot_no_flops_gl2_time', 'ot_no_flops_hbm_time', 'ot_no_flops_valu_time', 'ot_no_flops_mfma_time', 
    'ot_under_util_lds_time', 'ot_under_util_gl1_time', 'ot_under_util_gl2_time', 'ot_under_util_hbm_time', 'ot_under_util_valu_time', 'ot_under_util_mfma_time', 
    'ot_above_util_lds_time', 'ot_above_util_gl1_time', 'ot_above_util_gl2_time', 'ot_above_util_hbm_time', 'ot_above_util_valu_time', 'ot_above_util_mfma_time'
]

# Lastly, output all results into a CSV file.
with open(OUTPUT_FILE_NAME, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(output_headers)
    for gpu_id in ot_gpu_ids:
        output_row = []
        output_row.append(gpu_id)
        output_row.append(ot_total_time)
        if (ot_gpu_time_map):
            output_row.append(sum(ot_gpu_time_map[gpu_id]))
        else:
            output_row.append('0')
        if (ot_host_device_time_map):
            output_row.append(sum(ot_host_device_time_map[gpu_id]))
        else:
            output_row.append('0')
        if(ot_device_time_host_map):
            output_row.append(sum(ot_device_time_host_map[gpu_id]))
        else:
            output_row.append('0')
        if(ot_gpu_gpu_comm_time_map):
            output_row.append(sum(ot_gpu_gpu_comm_time_map[gpu_id]))
        else:
            output_row.append('0')
        if (ot_gpu_invoke_time_map):
            output_row.append(sum(ot_gpu_invoke_time_map[gpu_id]))
        else:
            output_row.append('0')
        op_no_flops_time, op_below_util_time, op_above_util_time = op_gpu_bounds_time_map[gpu_id]
        op_gpu_sum = op_gpu_gpu_comm_time_map[gpu_id] + \
            sum(op_no_flops_time) + sum(op_below_util_time) + sum(op_above_util_time)
        output_row.append(op_gpu_sum)
        output_row.append(op_gpu_gpu_comm_time_map[gpu_id])
        output_row.append(UTIL_THRESHOLD_RATIO)
        output_row.extend(op_no_flops_time)
        output_row.extend(op_below_util_time)
        output_row.extend(op_above_util_time)
        writer.writerow(output_row)
    
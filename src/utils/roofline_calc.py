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

from dataclasses import dataclass
from utils.utils import console_debug
import csv

################################################
# Global vars
################################################

IMGNAME = "empirRoof"

L2_BANKS = 32  # default assuming mi200

XMIN = 0.01
XMAX = 1000

FONT_SIZE = 16
FONT_COLOR = "black"
FONT_WEIGHT = "bold"

SUPPORTED_SOC = ["mi200"]

TOP_N = 10


################################################
# Helper funcs
################################################
@dataclass
class AI_Data:
    KernelName: str
    numCalls: float

    total_flops: float
    valu_flops: float
    mfma_flops_f16: float
    mfma_flops_bf16: float
    mfma_flops_f32: float
    mfma_flops_f64: float
    mfma_iops_i8: float
    lds_data: float
    L1cache_data: float
    L2cache_data: float
    hbm_data: float

    totalDuration: float
    avgDuration: float


def get_font():
    return {
        "size": FONT_SIZE,
        "color": FONT_COLOR,
        "weight": FONT_WEIGHT,
        "family": "serif",
    }


def get_color(catagory):
    if catagory == "ai_l1":
        return "green"
    elif catagory == "ai_l2":
        return "blue"
    elif catagory == "ai_hbm":
        return "red"
    else:
        raise RuntimeError("Invalid catagory passed to get_color()")


# -------------------------------------------------------------------------------------
#                           Plot BW at each cache level
# -------------------------------------------------------------------------------------
def calc_ceilings(roofline_parameters, dtype, benchmark_data):
    """Given benchmarking data, calculate ceilings (or peak performance) for empirical roofline"""
    # TODO: This is where filtering by memory level will need to occur for standalone
    graphPoints = {"hbm": [], "l2": [], "l1": [], "lds": [], "valu": [], "mfma": []}

    if roofline_parameters["mem_level"] == "ALL":
        cacheHierarchy = ["HBM", "L2", "L1", "LDS"]
    else:
        cacheHierarchy = roofline_parameters["mem_level"]

    x1 = y1 = x2 = y2 = -1
    x1_mfma = y1_mfma = x2_mfma = y2_mfma = -1
    target_precision = dtype[2:]

    if dtype != "FP16" and dtype != "I8":
        peakOps = float(benchmark_data[dtype + "Flops"][roofline_parameters["device_id"]])
    for i in range(0, len(cacheHierarchy)):
        # Plot BW line
        console_debug("roofline", "Current cache level is %s" % cacheHierarchy[i])
        curr_bw = cacheHierarchy[i] + "Bw"
        peakBw = float(benchmark_data[curr_bw][roofline_parameters["device_id"]])

        if dtype == "I8":
            peakMFMA = float(
                benchmark_data["MFMAI8Ops"][roofline_parameters["device_id"]]
            )
        else:
            peakMFMA = float(
                benchmark_data["MFMAF{}Flops".format(target_precision)][
                    roofline_parameters["device_id"]
                ]
            )

        x1 = float(XMIN)
        y1 = float(XMIN) * peakBw
        # Note: No reg peakOps for FP16 or INT8
        if dtype != "FP16" and dtype != "I8":
            x2 = peakOps / peakBw
            y2 = peakOps

            # Plot MFMA lines (NOTE: Assuming MI200 soc)
            x1_mfma = peakOps / peakBw
            y1_mfma = peakOps

        x2_mfma = peakMFMA / peakBw
        y2_mfma = peakMFMA

        # These are the points to use:
        console_debug("roofline", "coordinate points:")
        console_debug("x = [{}, {}]".format(x1, x2_mfma))
        console_debug("y = [{}, {}]".format(y1, y2_mfma))

        graphPoints[cacheHierarchy[i].lower()].append([x1, x2_mfma])
        graphPoints[cacheHierarchy[i].lower()].append([y1, y2_mfma])
        graphPoints[cacheHierarchy[i].lower()].append(peakBw)

    # -------------------------------------------------------------------------------------
    #                                     Plot computing roof
    # -------------------------------------------------------------------------------------
    # Note: No FMA roof for FP16 or INT8
    if dtype != "FP16" and dtype != "I8":
        # Plot FMA roof
        x0 = XMAX
        if x2 < x0:
            x0 = x2

        console_debug("FMA ROOF [{}, {}], [{},{}]".format(x0, XMAX, peakOps, peakOps))
        graphPoints["valu"].append([x0, XMAX])
        graphPoints["valu"].append([peakOps, peakOps])
        graphPoints["valu"].append(peakOps)

    # Plot MFMA roof
    if (
        x1_mfma != -1 or dtype == "FP16" or dtype == "I8"
    ):  # assert that mfma has been assigned
        x0_mfma = XMAX
        if x2_mfma < x0_mfma:
            x0_mfma = x2_mfma

        console_debug(
            "MFMA ROOF [{}, {}], [{},{}]".format(x0_mfma, XMAX, peakMFMA, peakMFMA)
        )
        graphPoints["mfma"].append([x0_mfma, XMAX])
        graphPoints["mfma"].append([peakMFMA, peakMFMA])
        graphPoints["mfma"].append(peakMFMA)

    return graphPoints


# -------------------------------------------------------------------------------------
#                              Overlay application performance
# -------------------------------------------------------------------------------------
# Calculate relevant metrics for ai calculation
def calc_ai(sort_type, ret_df):
    """Given counter data, calculate arithmetic intensity for each kernel in the application."""
    df = ret_df["pmc_perf"]
    # Sort by top kernels or top dispatches?
    df = df.sort_values(by=["Kernel_Name"])
    df = df.reset_index(drop=True)

    total_flops = valu_flops = mfma_flops_bf16 = mfma_flops_f16 = mfma_iops_i8 = (
        mfma_flops_f32
    ) = mfma_flops_f64 = lds_data = L1cache_data = L2cache_data = hbm_data = calls = (
        totalDuration
    ) = avgDuration = 0.0

    kernelName = ""

    myList = []
    at_end = False
    next_kernelName = ""

    for idx in df.index:
        # CASE: Top kernels
        # Calculate + append AI data if
        # a) current KernelName is different than previous OR
        # b) We've reached the end of list
        if idx + 1 == df.shape[0]:
            at_end = True
        else:
            next_kernelName = df["Kernel_Name"][idx + 1]

        kernelName = df["Kernel_Name"][idx]
        try:
            total_flops += (
                (
                    64
                    * (
                        df["SQ_INSTS_VALU_ADD_F16"][idx]
                        + df["SQ_INSTS_VALU_MUL_F16"][idx]
                        + (2 * df["SQ_INSTS_VALU_FMA_F16"][idx])
                        + df["SQ_INSTS_VALU_TRANS_F16"][idx]
                    )
                )
                + (
                    64
                    * (
                        df["SQ_INSTS_VALU_ADD_F32"][idx]
                        + df["SQ_INSTS_VALU_MUL_F32"][idx]
                        + (2 * df["SQ_INSTS_VALU_FMA_F32"][idx])
                        + df["SQ_INSTS_VALU_TRANS_F32"][idx]
                    )
                )
                + (
                    64
                    * (
                        df["SQ_INSTS_VALU_ADD_F64"][idx]
                        + df["SQ_INSTS_VALU_MUL_F64"][idx]
                        + (2 * df["SQ_INSTS_VALU_FMA_F64"][idx])
                        + df["SQ_INSTS_VALU_TRANS_F64"][idx]
                    )
                )
                + (df["SQ_INSTS_VALU_MFMA_MOPS_F16"][idx] * 512)
                + (df["SQ_INSTS_VALU_MFMA_MOPS_BF16"][idx] * 512)
                + (df["SQ_INSTS_VALU_MFMA_MOPS_F32"][idx] * 512)
                + (df["SQ_INSTS_VALU_MFMA_MOPS_F64"][idx] * 512)
            )
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped total_flops at index {}".format(kernelName[:35], idx),
            )
            pass
        try:
            valu_flops += (
                64
                * (
                    df["SQ_INSTS_VALU_ADD_F16"][idx]
                    + df["SQ_INSTS_VALU_MUL_F16"][idx]
                    + (2 * df["SQ_INSTS_VALU_FMA_F16"][idx])
                    + df["SQ_INSTS_VALU_TRANS_F16"][idx]
                )
                + 64
                * (
                    df["SQ_INSTS_VALU_ADD_F32"][idx]
                    + df["SQ_INSTS_VALU_MUL_F32"][idx]
                    + (2 * df["SQ_INSTS_VALU_FMA_F32"][idx])
                    + df["SQ_INSTS_VALU_TRANS_F32"][idx]
                )
                + 64
                * (
                    df["SQ_INSTS_VALU_ADD_F64"][idx]
                    + df["SQ_INSTS_VALU_MUL_F64"][idx]
                    + (2 * df["SQ_INSTS_VALU_FMA_F64"][idx])
                    + df["SQ_INSTS_VALU_TRANS_F64"][idx]
                )
            )
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped valu_flops at index {}".format(kernelName[:35], idx),
            )
            pass

        try:
            mfma_flops_f16 += df["SQ_INSTS_VALU_MFMA_MOPS_F16"][idx] * 512
            mfma_flops_bf16 += df["SQ_INSTS_VALU_MFMA_MOPS_BF16"][idx] * 512
            mfma_flops_f32 += df["SQ_INSTS_VALU_MFMA_MOPS_F32"][idx] * 512
            mfma_flops_f64 += df["SQ_INSTS_VALU_MFMA_MOPS_F64"][idx] * 512
            mfma_iops_i8 += df["SQ_INSTS_VALU_MFMA_MOPS_I8"][idx] * 512
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped mfma ops at index {}".format(kernelName[:35], idx),
            )
            pass

        try:
            lds_data += (
                (df["SQ_LDS_IDX_ACTIVE"][idx] - df["SQ_LDS_BANK_CONFLICT"][idx])
                * 4
                * L2_BANKS
            )  # L2_BANKS = 32 (since assuming mi200)
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped lds_data at index {}".format(kernelName[:35], idx),
            )
            pass

        try:
            L1cache_data += df["TCP_TOTAL_CACHE_ACCESSES_sum"][idx] * 64
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped L1cache_data at index {}".format(kernelName[:35], idx),
            )
            pass

        try:
            L2cache_data += (
                df["TCP_TCC_WRITE_REQ_sum"][idx] * 64
                + df["TCP_TCC_ATOMIC_WITH_RET_REQ_sum"][idx] * 64
                + df["TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum"][idx] * 64
                + df["TCP_TCC_READ_REQ_sum"][idx] * 64
            )
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped L2cache_data at index {}".format(kernelName[:35], idx),
            )
            pass
        try:
            hbm_data += (
                (df["TCC_EA_RDREQ_32B_sum"][idx] * 32)
                + ((df["TCC_EA_RDREQ_sum"][idx] - df["TCC_EA_RDREQ_32B_sum"][idx]) * 64)
                + (df["TCC_EA_WRREQ_64B_sum"][idx] * 64)
                + ((df["TCC_EA_WRREQ_sum"][idx] - df["TCC_EA_WRREQ_64B_sum"][idx]) * 32)
            )
        except KeyError:
            console_debug(
                "roofline",
                "{}: Skipped hbm_data at index {}".format(kernelName[:35], idx),
            )
            pass

        totalDuration += df["End_Timestamp"][idx] - df["Start_Timestamp"][idx]
        avgDuration += df["End_Timestamp"][idx] - df["Start_Timestamp"][idx]

        calls += 1

        if sort_type == "kernels" and (at_end == True or (kernelName != next_kernelName)):
            myList.append(
                AI_Data(
                    kernelName,
                    calls,
                    total_flops / calls,
                    valu_flops / calls,
                    mfma_flops_f16 / calls,
                    mfma_flops_bf16 / calls,
                    mfma_flops_f32 / calls,
                    mfma_flops_f64 / calls,
                    mfma_iops_i8 / calls,
                    lds_data / calls,
                    L1cache_data / calls,
                    L2cache_data / calls,
                    hbm_data / calls,
                    totalDuration,
                    avgDuration / calls,
                )
            )
            console_debug(
                "Just added {} to AI_Data at index {}. # of calls: {}".format(
                    kernelName, idx, calls
                )
            )
            total_flops = valu_flops = mfma_flops_bf16 = mfma_flops_f16 = mfma_iops_i8 = (
                mfma_flops_f32
            ) = mfma_flops_f64 = lds_data = L1cache_data = L2cache_data = hbm_data = (
                calls
            ) = totalDuration = avgDuration = 0.0

        if sort_type == "dispatches":
            myList.append(
                AI_Data(
                    kernelName,
                    calls,
                    total_flops,
                    valu_flops,
                    mfma_flops_f16,
                    mfma_flops_bf16,
                    mfma_flops_f32,
                    mfma_flops_f64,
                    mfma_iops_i8,
                    lds_data,
                    L1cache_data,
                    L2cache_data,
                    hbm_data,
                    totalDuration,
                    avgDuration,
                )
            )
            total_flops = valu_flops = mfma_flops_bf16 = mfma_flops_f16 = mfma_iops_i8 = (
                mfma_flops_f32
            ) = mfma_flops_f64 = lds_data = L1cache_data = L2cache_data = hbm_data = (
                calls
            ) = totalDuration = avgDuration = 0.0

    myList.sort(key=lambda x: x.totalDuration, reverse=True)

    # print("Top 5 intensities ('{}')...".format(roof_details["sort"]))
    intensities = {"ai_l1": [], "ai_l2": [], "ai_hbm": []}
    curr_perf = []
    kernelNames = []
    i = 0
    # Create list of top 5 intensities
    while i < TOP_N and i != len(myList):
        kernelNames.append(myList[i].KernelName)
        (
            intensities["ai_l1"].append(myList[i].total_flops / myList[i].L1cache_data)
            if myList[i].L1cache_data
            else intensities["ai_l1"].append(0)
        )
        # print("cur_ai_L1", myList[i].total_flops/myList[i].L1cache_data) if myList[i].L1cache_data else print("null")
        # print()
        (
            intensities["ai_l2"].append(myList[i].total_flops / myList[i].L2cache_data)
            if myList[i].L2cache_data
            else intensities["ai_l2"].append(0)
        )
        # print("cur_ai_L2", myList[i].total_flops/myList[i].L2cache_data) if myList[i].L2cache_data else print("null")
        # print()
        (
            intensities["ai_hbm"].append(myList[i].total_flops / myList[i].hbm_data)
            if myList[i].hbm_data
            else intensities["ai_hbm"].append(0)
        )
        # print("cur_ai_hbm", myList[i].total_flops/myList[i].hbm_data) if myList[i].hbm_data else print("null")
        # print()
        (
            curr_perf.append(myList[i].total_flops / myList[i].avgDuration)
            if myList[i].avgDuration
            else curr_perf.append(0)
        )
        # print("cur_perf", myList[i].total_flops/myList[i].avgDuration) if myList[i].avgDuration else print("null")

        i += 1

    intensityPoints = {"ai_l1": [], "ai_l2": [], "ai_hbm": []}

    for i in intensities:
        values = intensities[i]

        color = get_color(i)
        x = []
        y = []
        for entryIndx in range(0, len(values)):
            x.append(values[entryIndx])
            y.append(curr_perf[entryIndx])

        intensityPoints[i].append(x)
        intensityPoints[i].append(y)

    # Add an entry for kernel names
    intensityPoints["kernelNames"] = kernelNames

    return intensityPoints


def constuct_roof(roofline_parameters, dtype):
    benchmark_results = os.path.join(roofline_parameters["workload_dir"], "roofline.csv")
    # -----------------------------------------------------
    # Initialize roofline data dictionary from roofline.csv
    # -----------------------------------------------------
    benchmark_data = (
        {}
    )  # TODO: consider changing this to an ordered dict for consistency over py versions
    headers = []
    try:
        with open(benchmark_results, "r") as csvfile:
            csvReader = csv.reader(csvfile, delimiter=",")
            rowCount = 0
            for row in csvReader:
                row.pop(0)  # remove devID
                if rowCount == 0:
                    headers = row
                    for i in headers:
                        benchmark_data[i] = []
                else:
                    for i, key in enumerate(headers):
                        benchmark_data[key].append(row[i])

                rowCount += 1
        csvfile.close()
    except:
        graphPoints = {
            "hbm": [None, None, None],
            "l2": [None, None, None],
            "l1": [None, None, None],
            "lds": [None, None, None],
            "valu": [None, None, None],
            "mfma": [None, None, None],
        }
        return graphPoints

    # ------------------
    #  Generate Roofline
    # ------------------
    results = calc_ceilings(roofline_parameters, dtype, benchmark_data)

    return results

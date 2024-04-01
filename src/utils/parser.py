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

import ast
import astunparse
import re
import os
import warnings
import pandas as pd
import numpy as np
from utils import schema
from utils.utils import console_warning, console_error, demarcate
from pathlib import Path

# ------------------------------------------------------------------------------
# Internal global definitions

# NB:
# Ammolite is unique gemstone from the Rocky Mountains.
# "ammolite__" is a special internal prefix to mark build-in global variables
# calculated or parsed from raw data sources. Its range is only in this file.
# Any other general prefixes string, like "buildin__", might be used by the
# editor. Whenever change it to a new one, replace all appearances in this file.

# 001 is ID of pmc_kernel_top.csv table
pmc_kernel_top_table_id = 1

# Build-in $denom defined in mongodb query:
#       "denom": {
#              "$switch" : {
#                 "branches": [
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Wave"]} ,
#                         "then":  "&SQ_WAVES"
#                    },
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Cycle"]} ,
#                         "then":  "&GRBM_GUI_ACTIVE"
#                    },
#                    {
#                         "case":  { "$eq": [ $normUnit, "per Sec"]} ,
#                         "then":  {"$divide":[{"$subtract": ["&End_Timestamp", "&Start_Timestamp" ]}, 1000000000]}
#                    }
#                 ],
#                "default": 1
#              }
#       }
supported_denom = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    "per_kernel": "1",
}

# Build-in defined in mongodb variables:
build_in_vars = {
    "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
    "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
    "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
    "numActiveCUs": "TO_INT(MIN((((ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / $GRBM_GUI_ACTIVE_PER_XCD)), \
              0) / $max_waves_per_cu) * 8) + MIN(MOD(ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) \
              / $GRBM_GUI_ACTIVE_PER_XCD)), 0), $max_waves_per_cu), 8)), $cu_per_gpu))",
    "kernelBusyCycles": "ROUND(AVG((((End_Timestamp - Start_Timestamp) / 1000) * $max_sclk)), 0)",
}

supported_call = {
    # If the below has single arg, like(expr), it is a aggr, in which turn to a pd function.
    # If it has args like list [], in which turn to a python function.
    "MIN": "to_min",
    "MAX": "to_max",
    # simple aggr
    "AVG": "to_avg",
    "MEDIAN": "to_median",
    "STD": "to_std",
    # functions apply to whole column of df or a single value
    "TO_INT": "to_int",
    # Support the below with 2 inputs
    "ROUND": "to_round",
    "QUANTILE": "to_quantile",
    "MOD": "to_mod",
    # Concat operation from the memory chart "active cus"
    "CONCAT": "to_concat",
}

# ------------------------------------------------------------------------------


def to_min(*args):
    if len(args) == 1 and isinstance(args[0], pd.core.series.Series):
        return args[0].min()
    elif min(args) == None:
        return np.nan
    else:
        return min(args)


def to_max(*args):
    if len(args) == 1 and isinstance(args[0], pd.core.series.Series):
        return args[0].max()
    elif len(args) == 2 and (
        isinstance(args[0], pd.core.series.Series)
        or isinstance(args[1], pd.core.series.Series)
    ):
        return np.maximum(args[0], args[1])
    elif max(args) == None:
        return np.nan
    else:
        return max(args)


def to_avg(a):
    if str(type(a)) == "<class 'NoneType'>":
        return np.nan
    elif a.empty:
        return np.nan
    elif isinstance(a, pd.core.series.Series):
        return a.mean()
    else:
        raise Exception("to_avg: unsupported type.")


def to_median(a):
    if a is None:
        return None
    elif isinstance(a, pd.core.series.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return a.median()
    else:
        raise Exception("to_median: unsupported type.")


def to_std(a):
    if isinstance(a, pd.core.series.Series):
        return a.std()
    else:
        raise Exception("to_std: unsupported type.")


def to_int(a):
    if str(type(a)) == "<class 'NoneType'>":
        return np.nan
    elif isinstance(a, (int, float, np.int64)):
        return int(a)
    elif isinstance(a, pd.core.series.Series):
        return a.astype("Int64")
    # Do we need it?
    # elif isinstance(a, str):
    #     return int(a)
    else:
        raise Exception("to_int: unsupported type.")


def to_round(a, b):
    if isinstance(a, pd.core.series.Series):
        return a.round(b)
    else:
        return round(a, b)


def to_quantile(a, b):
    if a is None:
        return None
    elif isinstance(a, pd.core.series.Series):
        return a.quantile(b)
    else:
        raise Exception("to_quantile: unsupported type.")


def to_mod(a, b):
    if isinstance(a, pd.core.series.Series):
        return a.mod(b)
    else:
        return a % b


def to_concat(a, b):
    return str(a) + str(b)


class CodeTransformer(ast.NodeTransformer):
    """
    Python AST visitor to transform user defined equation string to df format
    """

    def visit_Call(self, node):
        self.generic_visit(node)
        # print("--- debug visit_Call --- ", node.args, node.func)
        # print(astunparse.dump(node))
        # print(astunparse.unparse(node))
        if isinstance(node.func, ast.Name):
            if node.func.id in supported_call:
                node.func.id = supported_call[node.func.id]
            else:
                raise Exception(
                    "Unknown call:", node.func.id
                )  # Could be removed if too strict
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        # print("visit_IfExp", type(node.test), type(node.body), type(node.orelse), dir(node))

        if isinstance(node.body, ast.Num):
            raise Exception(
                "Don't support body of IF with number only! Has to be expr with df['column']."
            )

        new_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=node.body, attr="where", ctx=ast.Load()),
                args=[node.test, node.orelse],
                keywords=[],
            )
        )
        # print("-------------")
        # print(astunparse.dump(new_node))
        # print("-------------")

        return new_node

    # NB:
    # visit_Name is for replacing HW counter to its df expr. In this way, we
    # could support any HW counter names, which is easier than regex.
    #
    # There are 2 limitations:
    #   - It is not straightforward to support types other than simple column
    #     in df, such as [], (). If we need to support those, have to implement
    #     in correct way or work around.
    #   - The 'raw_pmc_df' is hack code. For other data sources, like wavefront
    #     data,We need to think about template or pass it as a parameter.
    def visit_Name(self, node):
        self.generic_visit(node)
        # print("-------------", node.id)
        if (not node.id.startswith("ammolite__")) and (not node.id in supported_call):
            new_node = ast.Subscript(
                value=ast.Name(id="raw_pmc_df", ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s=node.id)),
                ctx=ast.Load(),
            )

            node = new_node
        return node


def build_eval_string(equation, coll_level):
    """
    Convert user defined equation string to eval executable string
    For example,
        input: AVG(100  * SQ_ACTIVE_INST_SCA / ( GRBM_GUI_ACTIVE * $numCU ))
        output: to_avg(100 * raw_pmc_df["pmc_perf"]["SQ_ACTIVE_INST_SCA"] / \
                 (raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] * numCU))
        input: AVG(((TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31) if (TCC_EA_RDREQ_31 != 0) else (0)))
        output: to_avg((raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_LEVEL_31"] / raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"]).where(raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"] != 0, 0))
        We can not handle the below for now,
        input: AVG((0 if (TCC_EA_RDREQ_31 == 0) else (TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31)))
        But potential workaound is,
        output: to_avg(raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"].where(raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"] == 0, raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_LEVEL_31"] / raw_pmc_df["pmc_perf"]["TCC_EA_RDREQ_31"]))
    """

    if coll_level is None:
        raise Exception("Error: coll_level can not be None.")

    if not equation:
        return ""

    s = str(equation)
    # print("input:", s)

    # build-in variable starts with '$', python can not handle it.
    # replace '$' with 'ammolite__'.
    # TODO: pre-check there is no "ammolite__" in all config files.
    s = re.sub(r"\$", "ammolite__", s)

    # convert equation string to intermediate expression in df array format
    ast_node = ast.parse(s)
    # print(astunparse.dump(ast_node))
    transformer = CodeTransformer()
    transformer.visit(ast_node)

    s = astunparse.unparse(ast_node)

    # correct column name/label in df with [], such as TCC_HIT[0],
    # the target is df['TCC_HIT[0]']
    s = re.sub(r"\'\]\[(\d+)\]", r"[\g<1>]']", s)
    # use .get() to catch any potential KeyErrors
    s = re.sub(r"raw_pmc_df\['(.*?)']", r'raw_pmc_df.get("\1")', s)
    # apply coll_level
    s = re.sub(r"raw_pmc_df", "raw_pmc_df.get('" + coll_level + "')", s)
    # print("--- build_eval_string, return: ", s)
    return s


def update_denom_string(equation, unit):
    """
    Update $denom in equation with runtime normalization unit.
    """
    if not equation:
        return ""

    s = str(equation)

    if unit in supported_denom.keys():
        s = re.sub(r"\$denom", supported_denom[unit], s)

    return s


def update_normUnit_string(equation, unit):
    """
    Update $normUnit in equation with runtime normalization unit.
    It is string replacement for display only.
    """

    # TODO: We might want to do it for subtitle contains $normUnit
    if not equation:
        return ""

    return re.sub(
        r"\((?P<PREFIX>\w*)\s+\+\s+(\$normUnit\))",
        r"\g<PREFIX> " + re.sub("_", " ", unit),
        str(equation),
    ).capitalize()


def gen_counter_list(formula):
    function_filter = {
        "MIN": None,
        "MAX": None,
        "AVG": None,
        "ROUND": None,
        "TO_INT": None,
        "GB": None,
        "STD": None,
        "GFLOP": None,
        "GOP": None,
        "OP": None,
        "CU": None,
        "NC": None,
        "UC": None,
        "CC": None,
        "RW": None,
        "GIOP": None,
        "GFLOPs": None,
        "CONCAT": None,
        "MOD": None,
    }

    built_in_counter = [
        "LDS_Per_Workgroup",
        "Grid_Size",
        "Workgroup_Size",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Scratch_Per_Workitem",
        "Start_Timestamp",
        "End_Timestamp",
    ]

    visited = False
    counters = []
    if not isinstance(formula, str):
        return visited, counters
    try:
        tree = ast.parse(
            formula.replace("$normUnit", "SQ_WAVES")
            .replace("$denom", "SQ_WAVES")
            .replace(
                "$numActiveCUs",
                "TO_INT(MIN((((ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) / $GRBM_GUI_ACTIVE_PER_XCD})), \
              0) / $maxWavesPerCU) * 8) + MIN(MOD(ROUND(AVG(((4 * SQ_BUSY_CU_CYCLES) \
              / $GRBM_GUI_ACTIVE_PER_XCD)), 0), $maxWavesPerCU), 8)), $numCU))",
            )
            .replace("$", "")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                val = str(node.id)[:-4] if str(node.id).endswith("_sum") else str(node.id)
                if val.isupper() and val not in function_filter:
                    counters.append(val)
                    visited = True
                if val in built_in_counter:
                    visited = True
    except:
        pass

    return visited, counters


def calc_builtin_var(var, sys_info):
    """
    Calculate build-in variable based on sys_info:
    """
    if isinstance(var, int):
        return var
    elif isinstance(var, str) and var.startswith("$total_l2_chan"):
        return sys_info.total_l2_chan
    else:
        console_error('Built-in var " %s " is not supported' % var)


@demarcate
def build_dfs(archConfigs, filter_metrics, sys_info):
    """
    - Build dataframe for each type of data source within each panel.
      Each dataframe will be used as a template to load data with each run later.
      For now, support "metric_table" and "raw_csv_table". Otherwise, put an empty df.
    - Collect/build metric_list to suport customrized metrics profiling.
    """

    # TODO: more error checking for filter_metrics!!
    # if filter_metrics:
    #     for metric in filter_metrics:
    #         if not metric in avail_ip_blocks:
    #             print("{} is not a valid metric to filter".format(metric))
    #             exit(1)
    simple_box = {
        "Min": ["MIN(", ")"],
        "Q1": ["QUANTILE(", ", 0.25)"],
        "Median": ["MEDIAN(", ")"],
        "Q3": ["QUANTILE(", ", 0.75)"],
        "Max": ["MAX(", ")"],
    }

    d = {}
    metric_list = {}
    dfs_type = {}
    metric_counters = {}
    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, data_config in data_source.items():
                if (
                    type == "metric_table"
                    and "metric" in data_config
                    and "placeholder_range" in data_config["metric"]
                ):
                    # print(data_config["metric"])
                    new_metrics = {}
                    # NB: support single placeholder for now!!
                    p_range = data_config["metric"].pop("placeholder_range")
                    metric, metric_expr = data_config["metric"].popitem()
                    # print(len(data_config["metric"]))
                    # data_config['metric'].clear()
                    for p, r in p_range.items():
                        # NB: We have to resolve placeholder range first if it
                        #   is a build-in var. It will be too late to do it in
                        #   eval_metric(). This is the only reason we need
                        #   sys_info at this stage.
                        var = calc_builtin_var(r, sys_info)
                        for i in range(var):
                            new_key = metric.replace(p, str(i))
                            new_val = {}
                            for k, v in metric_expr.items():
                                new_val[k] = metric_expr[k].replace(p, str(i))
                            # print(new_val)
                            new_metrics[new_key] = new_val

                    # print(p_range)
                    # print(new_metrics)
                    data_config["metric"] = new_metrics
                    # print(data_config)
                    # print(data_config["metric"])

    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, data_config in data_source.items():
                if type == "metric_table":
                    headers = ["Metric_ID"]
                    data_source_idx = str(data_config["id"] // 100)
                    if data_source_idx != 0 or data_source_idx in filter_metrics:
                        metric_list[data_source_idx] = panel["title"]
                    if (
                        "cli_style" in data_config
                        and data_config["cli_style"] == "simple_box"
                    ):
                        headers.append(data_config["header"]["metric"])
                        for k in simple_box.keys():
                            headers.append(k)

                        for key, tile in data_config["header"].items():
                            if key != "metric" and key != "tips" and key != "expr":
                                headers.append(tile)
                    else:
                        for key, tile in data_config["header"].items():
                            if key != "tips":
                                headers.append(tile)

                    # do we always need one?
                    headers.append("coll_level")
                    if "tips" in data_config["header"].keys():
                        headers.append(data_config["header"]["tips"])

                    df = pd.DataFrame(columns=headers)

                    i = 0
                    for key, entries in data_config["metric"].items():
                        data_source_idx = (
                            str(data_config["id"] // 100)
                            + "."
                            + str(data_config["id"] % 100)
                        )
                        metric_idx = data_source_idx + "." + str(i)
                        values = []
                        eqn_content = []

                        if (
                            (not filter_metrics)
                            or (
                                metric_idx in filter_metrics
                            )  # no filter  # metric in filter
                            or
                            # the whole table in filter
                            (data_source_idx in filter_metrics)
                            or
                            # the whole IP block in filter
                            (str(panel_id // 100) in filter_metrics)
                        ):
                            values.append(metric_idx)
                            values.append(key)

                            metric_list[data_source_idx] = data_config["title"]

                            if (
                                "cli_style" in data_config
                                and data_config["cli_style"] == "simple_box"
                            ):
                                # print("~~~~~~~~~~~~~~~~~")
                                # print(entries)
                                # print("~~~~~~~~~~~~~~~~~")
                                for k, v in entries.items():
                                    if k == "expr":
                                        for bk, bv in simple_box.items():
                                            values.append(bv[0] + v + bv[1])
                                    else:
                                        if (
                                            k != "tips"
                                            and k != "coll_level"
                                            and k != "alias"
                                        ):
                                            values.append(v)

                            else:
                                for k, v in entries.items():
                                    if k != "tips" and k != "coll_level" and k != "alias":
                                        values.append(v)
                                        eqn_content.append(v)

                            if "alias" in entries.keys():
                                values.append(entries["alias"])

                            if "coll_level" in entries.keys():
                                values.append(entries["coll_level"])
                            else:
                                values.append(schema.pmc_perf_file_prefix)

                            if "tips" in entries.keys():
                                values.append(entries["tips"])

                            # print(headers, values)
                            # print(key, entries)
                            df_new_row = pd.DataFrame([values], columns=headers)
                            df = pd.concat([df, df_new_row])

                        # collect metric_list
                        metric_list[metric_idx] = key
                        # generate mapping of counters and metrics
                        filter = {}
                        _visited = False
                        for formula in eqn_content:
                            if formula is not None and formula != "None":
                                visited, counters = gen_counter_list(formula)
                                if visited:
                                    _visited = True
                                for k in counters:
                                    filter[k] = None

                        if len(filter) > 0 or _visited:
                            metric_counters[key] = list(filter)

                        i += 1

                    df.set_index("Metric_ID", inplace=True)
                    # df.set_index('Metric', inplace=True)
                    # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
                elif type == "raw_csv_table":
                    data_source_idx = str(data_config["id"] // 100)
                    if (
                        (not filter_metrics)
                        or (data_source_idx == "0")  # no filter
                        or (data_source_idx in filter_metrics)
                    ):
                        if (
                            "columnwise" in data_config
                            and data_config["columnwise"] == True
                        ):
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv_columnwise"]
                            )
                        else:
                            df = pd.DataFrame(
                                [data_config["source"]], columns=["from_csv"]
                            )
                        metric_list[data_source_idx] = panel["title"]
                    else:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

                d[data_config["id"]] = df
                dfs_type[data_config["id"]] = type

    setattr(archConfigs, "dfs", d)
    setattr(archConfigs, "metric_list", metric_list)
    setattr(archConfigs, "dfs_type", dfs_type)
    setattr(archConfigs, "metric_counters", metric_counters)


def build_metric_value_string(dfs, dfs_type, normal_unit):
    """
    Apply the real eval string to its field in the metric_table df.
    """

    for id, df in dfs.items():
        if dfs_type[id] == "metric_table":
            for expr in df.columns:
                if expr in schema.supported_field:
                    # NB: apply all build-in before building the whole string
                    df[expr] = df[expr].apply(update_denom_string, unit=normal_unit)

                    # NB: there should be a faster way to do with single apply
                    if not df.empty:
                        for i in range(df.shape[0]):
                            row_idx_label = df.index.to_list()[i]
                            # print(i, "row_idx_label", row_idx_label, expr)
                            if expr.lower() != "alias":
                                df.at[row_idx_label, expr] = build_eval_string(
                                    df.at[row_idx_label, expr],
                                    df.at[row_idx_label, "coll_level"],
                                )

                elif expr.lower() == "unit" or expr.lower() == "units":
                    df[expr] = df[expr].apply(update_normUnit_string, unit=normal_unit)

        # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))


@demarcate
def eval_metric(dfs, dfs_type, sys_info, raw_pmc_df, debug):
    """
    Execute the expr string for each metric in the df.
    """

    # confirm no illogical counter values (only consider non-roofline runs)
    roof_only_run = sys_info.ip_blocks == "roofline"
    rocscope_run = sys_info.ip_blocks == "rocscope"
    if (
        (not rocscope_run and not roof_only_run)
        and hasattr(raw_pmc_df["pmc_perf"], "GRBM_GUI_ACTIVE")
        and (raw_pmc_df["pmc_perf"]["GRBM_GUI_ACTIVE"] == 0).any()
    ):
        console_warning("Dectected GRBM_GUI_ACTIVE == 0")
        console_error("Hauting execution for warning above.")

    ammolite__se_per_gpu = sys_info.se_per_gpu
    ammolite__pipes_per_gpu = sys_info.pipes_per_gpu
    ammolite__cu_per_gpu = sys_info.cu_per_gpu
    ammolite__simd_per_cu = sys_info.simd_per_cu  # not used
    ammolite__sqc_per_gpu = sys_info.sqc_per_gpu
    ammolite__lds_banks_per_cu = sys_info.lds_banks_per_cu
    ammolite__cur_sclk = sys_info.cur_sclk  # not used
    ammolite__mclk = sys_info.cur_mclk  # not used
    ammolite__max_sclk = sys_info.max_sclk
    ammolite__max_waves_per_cu = sys_info.max_waves_per_cu
    ammolite__hbm_bw = sys_info.hbm_bw
    ammolite__total_l2_chan = calc_builtin_var("$total_l2_chan", sys_info)
    ammolite__num_xcd = sys_info.num_xcd

    # TODO: fix all $normUnit in Unit column or title

    # build and eval all derived build-in global variables
    ammolite__build_in = {}

    # first pass, we do all per-xcd values, as these are used in subsequent builtins
    for key, value in build_in_vars.items():
        if "PER_XCD" not in key:
            continue
        # NB: assume all built-in vars from pmc_perf.csv for now
        s = build_eval_string(value, schema.pmc_perf_file_prefix)
        try:
            ammolite__build_in[key] = eval(compile(s, "<string>", "eval"))
        except TypeError:
            ammolite__build_in[key] = None
        except AttributeError as ae:
            if ae == "'NoneType' object has no attribute 'get'":
                ammolite__build_in[key] = None
    ammolite__GRBM_GUI_ACTIVE_PER_XCD = ammolite__build_in["GRBM_GUI_ACTIVE_PER_XCD"]
    ammolite__GRBM_COUNT_PER_XCD = ammolite__build_in["GRBM_COUNT_PER_XCD"]
    ammolite__GRBM_SPI_BUSY_PER_XCD = ammolite__build_in["GRBM_SPI_BUSY_PER_XCD"]

    for key, value in build_in_vars.items():
        # next pass, we evaluate the builtins the depend on the per-XCD values
        if "PER_XCD" in key:
            continue
        # NB: assume all built-in vars from pmc_perf.csv for now
        s = build_eval_string(value, schema.pmc_perf_file_prefix)
        try:
            ammolite__build_in[key] = eval(compile(s, "<string>", "eval"))
        except TypeError:
            ammolite__build_in[key] = None
        except AttributeError as ae:
            if ae == "'NoneType' object has no attribute 'get'":
                ammolite__build_in[key] = None
    ammolite__numActiveCUs = ammolite__build_in["numActiveCUs"]
    ammolite__kernelBusyCycles = ammolite__build_in["kernelBusyCycles"]

    # Hmmm... apply + lambda should just work
    # df['Value'] = df['Value'].apply(lambda s: eval(compile(str(s), '<string>', 'eval')))
    for id, df in dfs.items():
        if dfs_type[id] == "metric_table":
            for idx, row in df.iterrows():
                for expr in df.columns:
                    if expr in schema.supported_field:
                        if expr.lower() != "alias":
                            if row[expr]:
                                if debug:  # debug won't impact the regular calc
                                    print("~" * 40 + "\nExpression:")
                                    print(expr, "=", row[expr])
                                    print("Inputs:")
                                    matched_vars = re.findall(r"ammolite__\w+", row[expr])
                                    if matched_vars:
                                        for v in matched_vars:
                                            print(
                                                "Var ",
                                                v,
                                                ":",
                                                eval(compile(v, "<string>", "eval")),
                                            )
                                    matched_cols = re.findall(
                                        r"raw_pmc_df\['\w+'\]\['\w+'\]", row[expr]
                                    )
                                    if matched_cols:
                                        for c in matched_cols:
                                            m = re.match(
                                                r"raw_pmc_df\['(\w+)'\]\['(\w+)'\]", c
                                            )
                                            t = raw_pmc_df[m.group(1)][
                                                m.group(2)
                                            ].to_list()
                                            print(c)
                                            print(
                                                raw_pmc_df[m.group(1)][
                                                    m.group(2)
                                                ].to_list()
                                            )
                                            # print(
                                            #     tabulate(raw_pmc_df[m.group(1)][
                                            #         m.group(2)],
                                            #              headers='keys',
                                            #              tablefmt='fancy_grid'))
                                    print("\nOutput:")
                                    try:
                                        print(
                                            eval(compile(row[expr], "<string>", "eval"))
                                        )
                                        print("~" * 40)
                                    except TypeError:
                                        console_warning(
                                            "Skipping entry. Encountered a missing counter\n{} has been assigned to None\n{}".format(
                                                expr, np.nan
                                            )
                                        )
                                    except AttributeError as ae:
                                        if (
                                            str(ae)
                                            == "'NoneType' object has no attribute 'get'"
                                        ):
                                            console_warning(
                                                "Skipping entry. Encountered a missing csv\n{}".format(
                                                    np.nan
                                                )
                                            )
                                        else:
                                            console_error("analysis", str(ae))

                                # print("eval_metric", id, expr)
                                try:
                                    out = eval(compile(row[expr], "<string>", "eval"))
                                    if row.name != "19.1.1" and np.isnan(
                                        out
                                    ):  # Special exception for unique format of Active CUs in mem chart
                                        row[expr] = ""
                                    else:
                                        row[expr] = out
                                except TypeError:
                                    row[expr] = ""
                                except AttributeError as ae:
                                    if (
                                        str(ae)
                                        == "'NoneType' object has no attribute 'get'"
                                    ):
                                        row[expr] = ""
                                    else:
                                        console_error("analysis", str(ae))

                            else:
                                # If not insert nan, the whole col might be treated
                                # as string but not nubmer if there is NONE
                                row[expr] = ""

            # print(tabulate(df, headers='keys', tablefmt='fancy_grid'))


@demarcate
def apply_filters(workload, dir, is_gui, debug):
    """
    Apply user's filters to the raw_pmc df.
    """

    # TODO: error out properly if filters out of bound
    ret_df = workload.raw_pmc

    if workload.filter_gpu_ids:
        ret_df = ret_df.loc[
            ret_df[schema.pmc_perf_file_prefix]["GPU_ID"]
            .astype(str)
            .isin([workload.filter_gpu_ids])
        ]
        if ret_df.empty:
            console_error(
                "analysis", "{} is an invalid gpu-id".format(workload.filter_gpu_ids)
            )

    # NB:
    # Kernel id is unique!
    # We pick up kernel names from kerne ids first.
    # Then filter valid entries with kernel names.
    if workload.filter_kernel_ids:
        if all(type(kid) == int for kid in workload.filter_kernel_ids):
            # Verify valid kernel filter
            kernels_df = pd.read_csv(os.path.join(dir, "pmc_kernel_top.csv"))
            for kernel_id in workload.filter_kernel_ids:
                if kernel_id >= len(kernels_df["Kernel_Name"]):
                    console_error(
                        "{} is an invalid kernel id. Please enter an id between 0-{}".format(
                            kernel_id, len(kernels_df["Kernel_Name"]) - 1
                        )
                    )
            kernels = []
            # NB: mark selected kernels with "*"
            #    Todo: fix it for unaligned comparison
            kernel_top_df = workload.dfs[pmc_kernel_top_table_id]
            kernel_top_df["S"] = ""
            for kernel_id in workload.filter_kernel_ids:
                # print("------- ", kernel_id)
                kernels.append(kernel_top_df.loc[kernel_id, "Kernel_Name"])
                kernel_top_df.loc[kernel_id, "S"] = "*"

            if kernels:
                # print("fitlered df:", len(df.index))
                ret_df = ret_df.loc[
                    ret_df[schema.pmc_perf_file_prefix]["Kernel_Name"].isin(kernels)
                ]
        elif all(type(kid) == str for kid in workload.filter_kernel_ids):
            df_cleaned = ret_df[schema.pmc_perf_file_prefix]["Kernel_Name"].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            ret_df = ret_df.loc[df_cleaned.isin(workload.filter_kernel_ids)]
        else:
            console_error(
                "analyze",
                "Mixing kernel indices and string filters is not currently supported",
            )

    if workload.filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        for d in workload.filter_dispatch_ids:
            if int(d) >= len(ret_df):  # subtract 2 bc of the two header rows
                console_error("analysis", "{} is an invalid dispatch id.".format(d))
        if ">" in workload.filter_dispatch_ids[0]:
            m = re.match(r"\> (\d+)", workload.filter_dispatch_ids[0])
            ret_df = ret_df[
                ret_df[schema.pmc_perf_file_prefix]["Dispatch_ID"] > int(m.group(1))
            ]
        else:
            dispatches = [int(x) for x in workload.filter_dispatch_ids]
            ret_df = ret_df.loc[dispatches]
    if debug:
        print("~" * 40, "\nraw pmc df info:\n")
        print(workload.raw_pmc.info())
        print("~" * 40, "\nfiltered pmc df info:")
        print(ret_df.info())

    return ret_df


@demarcate
def load_kernel_top(workload, dir):
    # NB:
    #   - Do pmc_kernel_top.csv loading before eval_metric because we need the kernel names.
    #   - There might be a better way/timing to load raw_csv_table.
    tmp = {}
    for id, df in workload.dfs.items():
        if "from_csv" in df.columns:
            file = Path.joinpath(Path(dir), df.loc[0, "from_csv"])
            if file.exists():
                tmp[id] = pd.read_csv(file)
            else:
                console_warning(
                    f"Couldn't load {file.name}. This may result in missing analysis data."
                )
        # NB: Special case for sysinfo. Probably room for improvement in this whole function design
        elif "from_csv_columnwise" in df.columns and id == 101:
            tmp[id] = workload.sys_info.transpose()
            # All transposed columns should be marked with a general header
            tmp[id].columns = ["Info"]
        elif "from_csv_columnwise" in df.columns:
            # NB:
            #   Another way might be doing transpose in tty like metric_table.
            #   But we need to figure out headers and comparison properly.
            file = Path.joinpath(Path(dir), df.loc[0, "from_csv_columnwise"])
            if file.exists():
                tmp[id] = pd.read_csv(file).transpose()
                # NB:
                #   All transposed columns should be marked with a general header,
                #   so tty could detect them and show them correctly in comparison.
                tmp[id].columns = ["Info"]
            else:
                console_warning(
                    f"Couldn't load {file.name}. This may result in missing analysis data."
                )
    workload.dfs.update(tmp)


@demarcate
def load_table_data(workload, dir, is_gui, debug, verbose, skipKernelTop=False):
    """
    Load data for all "raw_csv_table".
    Calculate mertric value for all "metric_table".
    """
    if not skipKernelTop:
        load_kernel_top(workload, dir)

    eval_metric(
        workload.dfs,
        workload.dfs_type,
        workload.sys_info.iloc[0],
        apply_filters(workload, dir, is_gui, debug),
        debug,
    )


def build_comparable_columns(time_unit):
    """
    Build comparable columns/headers for display
    """
    comparable_columns = schema.supported_field
    top_stat_base = ["Count", "Sum", "Mean", "Median", "Standard Deviation"]

    for h in top_stat_base:
        comparable_columns.append(h + "(" + time_unit + ")")

    return comparable_columns


def correct_sys_info(mspec, specs_correction: dict):
    """
    Correct system spec items manually
    """
    # todo: more err checking for string specs_correction

    pairs = dict(re.findall(r"(\w+):\s*(\d+)", specs_correction))

    for k, v in pairs.items():
        if not hasattr(mspec, str(k)):
            console_error(
                "analyze",
                f"Invalid specs correction '{k}'. Please use --specs option to peak valid specs",
            )
        setattr(mspec, str(k), v)
    return mspec.get_class_members()

##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

import pandas as pd
from pathlib import Path
from tabulate import tabulate
import numpy as np

from omniperf_analyze.utils import schema, parser

hidden_columns = ["Tips", "coll_level"]
hidden_sections = [1900, 2000]


def smartUnits(df):
    for idx, row in df[df["Unit"] == "Gb/s"].items():
        for curr_metric in row:
            if "Metric" in df:
                curr_row = df[df["Metric"] == curr_metric]
                if not curr_row.empty:
                    # fix values
                    if "Value" in curr_row:
                        vals = curr_row["Value"].values
                        new_units = []
                        percent_diff = 0

                        # if baseline
                        if isinstance(vals[0], np.ndarray):
                            val_1 = curr_row["Value"].values[0][0]
                            baseline = curr_row["Value"].values[0][1].split()
                            val_2 = float(baseline[0])
                            percent_diff = baseline[1]
                            vals = np.array([val_1, val_2])

                        # calculate units
                        for val in vals:
                            if isinstance(
                                val,
                                float,
                            ):
                                if val < 0.001:
                                    new_units.append("Kb/s")
                                elif val < 1:
                                    new_units.append("Mb/s")
                                    if len(new_units) == 2:
                                        if new_units[0] == "Kb/s":
                                            new_units[0] = "Mb/s"
                                else:
                                    new_units.append("Gb/s")
                                    if len(new_units) == 2:
                                        new_units[0] = "Gb/s"
                        if len(new_units) > 0:
                            # Convert to new_units
                            if new_units[0] == "Mb/s":
                                vals = 1000 * vals
                            elif new_units[0] == "Kb/s":
                                vals = 1000000 * vals
                            vals = vals.tolist()
                            # if baseline
                            if len(new_units) == 2:
                                vals[1] = str(vals[1]) + " " + str(percent_diff)

                            df.loc[df["Metric"] == curr_metric, "Value"] = vals
                            df.loc[df["Metric"] == curr_metric, "Unit"] = new_units[0]

                    elif "Avg" in curr_row:
                        avg_vals = curr_row["Avg"].values
                        max_vals = curr_row["Max"].values
                        min_vals = curr_row["Min"].values
                        new_units = []
                        percent_diff = 0

                        # if baseline
                        if isinstance(avg_vals[0], np.ndarray):
                            avg_baseline = curr_row["Avg"].values[0][1].split()
                            avg_percent_diff = avg_baseline[1]
                            avg_val_1 = curr_row["Avg"].values[0][0]
                            avg_val_2 = float(avg_baseline[0])
                            avg_vals = np.array([avg_val_1, avg_val_2])

                            min_baseline = curr_row["Min"].values[0][1].split()
                            min_percent_diff = min_baseline[1]
                            min_val_1 = curr_row["Min"].values[0][0]
                            min_val_2 = float(min_baseline[0])
                            min_vals = np.array([min_val_1, min_val_2])

                            max_baseline = curr_row["Max"].values[0][1].split()
                            max_percent_diff = max_baseline[1]
                            max_val_1 = curr_row["Max"].values[0][0]
                            max_val_2 = float(max_baseline[0])
                            max_vals = np.array([max_val_1, max_val_2])

                        # calculate units
                        for val in avg_vals:
                            if isinstance(
                                val,
                                float,
                            ):
                                if val < 0.001:
                                    new_units.append("Kb/s")
                                elif val < 1:
                                    new_units.append("Mb/s")
                                    if len(new_units) == 2:
                                        if new_units[0] == "Kb/s":
                                            new_units[0] = "Mb/s"
                                else:
                                    new_units.append("Gb/s")
                                    if len(new_units) == 2:
                                        new_units[0] = "Gb/s"
                        if len(new_units) > 0:
                            # Convert to new_units
                            if new_units[0] == "Mb/s":
                                avg_vals = 1000 * avg_vals
                                max_vals = 1000 * max_vals
                                min_vals = 1000 * min_vals
                            elif new_units[0] == "Kb/s":
                                avg_vals = 1000000 * avg_vals
                                max_vals = 1000000 * max_vals
                                min_vals = 1000000 * min_vals
                            avg_vals = avg_vals.tolist()
                            max_vals = max_vals.tolist()
                            min_vals = min_vals.tolist()

                        # if baseline
                        if len(new_units) == 2:
                            avg_vals[1] = str(avg_vals[1]) + " " + str(avg_percent_diff)
                            max_vals[1] = str(max_vals[1]) + " " + str(max_percent_diff)
                            min_vals[1] = str(min_vals[1]) + " " + str(min_percent_diff)

                        if len(new_units) > 0:
                            df.loc[df["Metric"] == curr_metric, "Avg"] = avg_vals
                            df.loc[df["Metric"] == curr_metric, "Max"] = max_vals
                            df.loc[df["Metric"] == curr_metric, "Min"] = min_vals
                            df.loc[df["Metric"] == curr_metric, "Unit"] = new_units[0]
        return df


def string_multiple_lines(source, width, max_rows):
    """
    Adjust string with multiple lines by inserting '\n'
    """
    idx = 0
    lines = []
    while idx < len(source) and len(lines) < max_rows:
        lines.append(source[idx : idx + width])
        idx += width

    if idx < len(source):
        last = lines[-1]
        lines[-1] = last[0:-3] + "..."
    return "\n".join(lines)


def show_all(args, runs, archConfigs, output):
    """
    Show all panels with their data in plain text mode.
    """
    comparable_columns = parser.build_comparable_columns(args.time_unit)

    for panel_id, panel in archConfigs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if panel_id in hidden_sections:
            continue
        ss = ""  # store content of all data_source from one pannel

        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                # take the 1st run as baseline
                base_run, base_data = next(iter(runs.items()))
                base_df = base_data.dfs[table_config["id"]]

                df = pd.DataFrame(index=base_df.index)

                for header in list(base_df.keys()):
                    if (
                        (not args.cols)
                        or (args.cols and base_df.columns.get_loc(header) in args.cols)
                        or (type == "raw_csv_table")
                    ):
                        if header in hidden_columns:
                            pass
                        elif header not in comparable_columns:
                            if (
                                type == "raw_csv_table"
                                and table_config["source"] == "pmc_kernel_top.csv"
                                and header == "KernelName"
                            ):
                                # NB: the width of kernel name might depend on the header of the table.
                                adjusted_name = base_df["KernelName"].apply(
                                    lambda x: string_multiple_lines(x, 40, 3)
                                )
                                df = pd.concat([df, adjusted_name], axis=1)
                            elif type == "raw_csv_table" and header == "Info":
                                for run, data in runs.items():
                                    cur_df = data.dfs[table_config["id"]]
                                    df = pd.concat([df, cur_df[header]], axis=1)
                            else:
                                df = pd.concat([df, base_df[header]], axis=1)
                        else:
                            for run, data in runs.items():
                                cur_df = data.dfs[table_config["id"]]
                                if (type == "raw_csv_table") or (
                                    type == "metric_table"
                                    and (not header in hidden_columns)
                                ):
                                    if run != base_run:
                                        # calc percentage over the baseline
                                        base_df[header] = [
                                            float(x) if x != "" else float(0)
                                            for x in base_df[header]
                                        ]
                                        # insert unit fix here
                                        cur_df[header] = [
                                            float(x) if x != "" else float(0)
                                            for x in cur_df[header]
                                        ]
                                        t_df = (
                                            pd.concat(
                                                [
                                                    base_df[header],
                                                    cur_df[header],
                                                ],
                                                axis=1,
                                            )
                                            .pct_change(axis="columns")
                                            .iloc[:, 1]
                                        )
                                        if args.verbose >= 2:
                                            print("---------", header, t_df)

                                        # show value + percentage
                                        # TODO: better alignment
                                        t_df = (
                                            cur_df[header]
                                            .astype(float)
                                            .round(args.decimal)
                                            .map(str)
                                            + " ("
                                            + t_df.astype(float)
                                            .mul(100)
                                            .round(args.decimal)
                                            .map(str)
                                            + "%)"
                                        )

                                        df = pd.concat([df, t_df], axis=1)
                                    else:
                                        # insert unit fix here
                                        cur_df[header] = [
                                            round(float(x), args.decimal)
                                            if x != ""
                                            else x
                                            for x in base_df[header]
                                        ]
                                        df = pd.concat([df, cur_df[header]], axis=1)
                if not df.empty:
                    if "Unit" in df:
                        df = smartUnits(df)
                    # subtitle for each table in a panel if existing
                    table_id_str = (
                        str(table_config["id"] // 100)
                        + "."
                        + str(table_config["id"] % 100)
                    )

                    if "title" in table_config and table_config["title"]:
                        ss += table_id_str + " " + table_config["title"] + "\n"

                    if args.df_file_dir:
                        p = Path(args.df_file_dir)
                        if not p.exists():
                            p.mkdir()
                        if p.is_dir():
                            if "title" in table_config and table_config["title"]:
                                table_id_str += "_" + table_config["title"]
                            df.to_csv(
                                p.joinpath(table_id_str.replace(" ", "_") + ".csv"),
                                index=False,
                            )

                    # NB:
                    # "columnwise: True" is a special attr of a table/df
                    # For raw_csv_table, such as system_info, we transpose the
                    # df when load it, because we need those items in column.
                    # For metric_table, we only need to show the data in column
                    # fash for now.
                    ss += (
                        tabulate(
                            df.transpose()
                            if type != "raw_csv_table"
                            and "columnwise" in table_config
                            and table_config["columnwise"] == True
                            else df,
                            headers="keys",
                            tablefmt="fancy_grid",
                            floatfmt="." + str(args.decimal) + "f",
                        )
                        + "\n"
                    )

        if ss:
            print("\n" + "-" * 80, file=output)
            print(str(panel_id // 100) + ". " + panel["title"], file=output)
            print(ss, file=output)


def show_kernels(args, runs, archConfigs, output):
    """
    Show the kernels from top stats.
    """
    print("\n" + "-" * 80, file=output)
    print("Detected Kernels", file=output)

    df = pd.DataFrame()
    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                for run, data in runs.items():
                    single_df = data.dfs[table_config["id"]]
                    # NB:
                    #   For pmc_kernel_top.csv, have to sort here if not
                    #   sorted when load_table_data.
                    df = pd.concat([df, single_df["KernelName"]], axis=1)

    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="fancy_grid",
            floatfmt="." + str(args.decimal) + "f",
        ),
        file=output,
    )

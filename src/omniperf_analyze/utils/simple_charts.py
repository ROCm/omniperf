##############################################################################bl
# MIT License
#
# Copyright (c) 2023 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

import math
from dataclasses import dataclass
import pandas as pd
import plotext as plt
import plotly.express as px


# Notes:
#   This file includes implementation of a few simple but common charts in CLI.
#   We try to auto-size the layout to cover most of the cases as default. If it
#   doesn't work, we could expose more controls for the ui designers with style
#   config in yaml for each dashboard.


def simple_bar(df, title=None):
    """
    Plot data with simple bar chart
    """

    # TODO: handle None properly

    if "Metric" in df.columns and "Value" in df.columns:
        metric_dict = (
            pd.DataFrame([df["Metric"], df["Value"]])
            .replace("", 0)
            .transpose()
            .set_index("Metric")
            .to_dict()["Value"]
        )
    else:
        raise NameError("simple_bar: No Metric or Value in df columns!")

    plt.clear_figure()

    # adjust plot size along x axis based on the max value
    w = max(list(metric_dict.values()))
    if w < 20 and w > 1:
        w *= 3
    elif w < 1:
        w *= 100
    plt.simple_bar(list(metric_dict.keys()), list(metric_dict.values()), width=w)
    # plt.show()
    return "\n" + plt.build() + "\n"


def simple_multiple_bar(df, title=None):
    """
    Plot data with simple multiple bar chart
    """

    # TODO: handle Nan and None properly

    plt.clear_figure()
    t_df = df.fillna(0).replace("", 0)
    sub_labels = t_df.transpose().to_dict("split")["index"]
    sub_labels.pop(0)
    data = t_df.transpose().to_dict("split")["data"]
    labels = data.pop(0)

    # plt.simple_multiple_bar(labels, data, labels = sub_labels) #, width=w)

    # print(data)
    plt.theme("pro")
    # adjust plot size along y axis based on the max value
    h = max(max(y) for y in data)
    # print(h)
    if h < 20 and h > 0.5:
        h *= 10
    elif h < 0.5 or math.isclose(h, 0.5):
        h *= 300

    plt.plot_size(height=h)
    plt.multiple_bar(labels, data, label=sub_labels, color=["blue", "blue+", 68, 63])

    # plt.show()
    return "\n" + plt.build() + "\n"


def simple_box(df, orientation="v", title=None):
    """
    Plot data with simple box/whisker chart.
    Accept pre-calculated data only for now.
    """

    # Example:
    # labels = ["apple", "bee", "cat", "dog"]
    # datas = [
    #     # max, q3, q2, q1, min
    #     [10, 7, 5, 3, 1.5],
    #     [19, 12.3, 9, 7, 4],
    #     [15, 14, 11, 9, 8],
    #     [13, 12, 11, 10, 6]]

    # plt.box(labels, datas, width=0.1, hint='hint')
    # plt.theme("pro")
    # plt.title("Most Favored Pizzas in the World")
    # plt.show()

    plt.clear_figure()
    labels = []
    data = []

    # TODO:
    # handle Nan and None properly
    # error checking for labels
    # show unit if provided

    labels_length = 0
    t_df = df.fillna(0).replace("", 0)
    for index, row in t_df.iterrows():
        labels.append(row["Metric"])
        labels_length += len(row["Metric"]) + 10
        data.append([row["Max"], row["Q3"], row["Median"], row["Q1"], row["Min"]])

    # print("~~~~~~~~~~~~~~~~~~~~")
    # print(labels)
    # print(labels_length)
    # print(data)
    # print("~~~~~~~~~~~~~~~~~~~~")
    # print(plt.bar.__doc__)

    if orientation == "v":
        # adjust plot size along x axis based on total labels length
        plt.plot_size(labels_length, 30)

    plt.box(
        labels,
        data,
        width=0.1,
        colors=["blue+", "orange+"],
        orientation=orientation,
        hint="hint",
    )
    plt.theme("pro")

    # plt.show()
    return "\n" + plt.build() + "\n"


def px_simple_bar(df, title: str = None, id=None, style: dict = None, orientation="h"):
    """
    Plot data with simple bar chart
    """

    # TODO: handle None properly
    if "Metric" in df.columns and ("Count" in df.columns or "Value" in df.columns):
        detected_label = "Count" if "Count" in df.columns else "Value"
        df[detected_label] = [
            x.astype(int) if x != "" else int(0) for x in df[detected_label]
        ]
    else:
        raise NameError("simple_bar: No Metric or Count in df columns!")

    # Assign figure characteristics
    range_color = style.get("range_color", None)
    label_txt = style.get("label_txt", None)
    xrange = style.get("xrange", None)
    if label_txt is not None:
        label_txt = label_txt.strip("()")
        try:
            label_txt = label_txt.replace("+ $normUnit", df["Unit"][0])
        except KeyError:
            print("No units found in df. Auto labeling.")

    # Overrides for figure chatacteristics
    if id == 1701.1:
        label_txt = "%"
        range_color = [0, 100]
        xrange = [0, 110]
    if id == 1701.2:
        label_txt = "Gb/s"
        range_color = [0, 1638]
        xrange = [0, 1638]

    fig = px.bar(
        df,
        title=title,
        x=detected_label,
        y="Metric",
        color=detected_label,
        range_color=range_color,
        labels={detected_label: label_txt},
        orientation=orientation,
    ).update_xaxes(range=xrange)

    return fig


def px_simple_multi_bar(df, title=None, id=None):
    """
    Plot data with simple multiple bar chart
    """

    # TODO: handle Nan and None properly
    if "Metric" in df.columns and "Avg" in df.columns:
        df["Avg"] = [x.astype(int) if x != "" else int(0) for x in df["Avg"]]
    else:
        raise NameError("simple_multi_bar: No Metric or Count in df columns!")

    dfigs = []
    nested_bar = {}
    df_unit = df["Unit"][0]
    if id == 1604:
        nested_bar = {"NC": {}, "UC": {}, "RW": {}, "CC": {}}
        for index, row in df.iterrows():
            nested_bar[row["Coherency"]][row["Xfer"]] = row["Avg"]
    if id == 1704:
        nested_bar = {"Read": {}, "Write": {}}
        for index, row in df.iterrows():
            nested_bar[row["Transaction"]][row["Type"]] = row["Avg"]

    for group, metric in nested_bar.items():
        dfigs.append(
            px.bar(
                title=group,
                x=metric.values(),
                y=metric.keys(),
                labels={"x": df_unit, "y": ""},
                text=metric.values(),
            )
            .update_xaxes(showgrid=False, rangemode="nonnegative")
            .update_yaxes(showgrid=False)
        )
    return dfigs

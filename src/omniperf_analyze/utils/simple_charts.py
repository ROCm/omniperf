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

from dataclasses import dataclass
import pandas as pd
import plotext as plt


def simple_bar(metric_dict, title=None):
    """
    Plot data with simple bar chart
    """

    # TODO: handle None properly

    plt.clear_figure()

    # adjust plot size along x axis based on the max value
    w = max(list(metric_dict.values()))
    if w < 20:
        w *= 3
    elif w < 1:
        w *= 100
    plt.simple_bar(list(metric_dict.keys()), list(metric_dict.values()), width=w)
    # plt.show()
    return "\n" + plt.build() + "\n"


def simple_box(df, orientation="v", title=None):
    """
    Plot data with simple box/whisker chart.
    Accept pre-calculated data only for now.
    """

    # Notes:
    #   We try to auto-size the layout to cover most of the cases as default.
    #   If it doesn't work, we could expose more controls for the ui designers
    #   with style config in yaml for each dashboard.

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
    # handle None properly
    # error checking for labels
    # show unit if provided

    labels_length = 0
    t_df = df.fillna(0)
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

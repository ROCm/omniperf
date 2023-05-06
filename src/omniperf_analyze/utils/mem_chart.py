##############################################################################bl
# MIT License
#
# Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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
##############################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Generator
from types import SimpleNamespace as NS
from plotille import *


# A basic rect frame for any block or group of wires where all its elements should
# be within this range, except the label(title) might be on the top of it.
@dataclass
class RectFrame:
    label: str
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 1.0
    y_max: float = 1.0


# Instr Buff Block
@dataclass
class InstrBuff(RectFrame):
    wave_occupancy: float = 0.0
    wave_life: float = 0.0

    def draw(self, canvas):
        #print("---------", self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        canvas.rect(self.x_min, self.y_min, self.x_max - 2.0, self.y_max - 1.0)
        canvas.rect(self.x_min + 1.0, self.y_min + 0.5, self.x_max - 1.0,
                    self.y_max - 0.5)
        canvas.rect(self.x_min + 2.0, self.y_min + 1.0, self.x_max, self.y_max)

        canvas.rect(self.x_min + 4.0, self.y_max - 3.5, self.x_max - 4.0,
                    self.y_max - 2.0)
        canvas.text(self.x_min + 5.0, self.y_max - 3.0, r"Wave   0 Instr Buf")

        canvas.rect(self.x_min + 4.0, self.y_max - 7.5, self.x_max - 4.0,
                    self.y_max - 6.0)
        canvas.text(self.x_min + 5.0, self.y_max - 7.0, r"Wave N-1 Instr Buf")

        canvas.text(self.x_min + 7.0, self.y_min + 5.0, r"Wave Occupancy")
        canvas.text(self.x_min + 10.0,
                    self.y_min + 4.0,
                    "{val:>3}  per CU".format(val=self.wave_occupancy),
                    color='yellow')
        canvas.text(self.x_min + 7.0, self.y_min + 3.0, r"Wave Life")
        canvas.text(self.x_min + 8.0,
                    self.y_min + 2.0,
                    "{val:>5}  cycles".format(val=self.wave_life),
                    color='yellow')


# Instr Dispatch Block
@dataclass
class InstrDispatch(RectFrame):
    top_rect_x_min: float = 0.0
    top_rect_x_max: float = 0.0
    top_rect_y_min: float = 0.0
    top_rect_y_max: float = 0.0
    text_x_offset: float = 1.0
    text_y_offset: float = 0.5
    line_y_offset: float = 0.5
    rect_y_offset: float = 3.0
    instrs: Dict[str, float] = field(default_factory=dict)

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        self.top_rect_x_min = self.x_min + 2.0
        self.top_rect_x_max = self.top_rect_x_min + 14.0
        self.top_rect_y_min = self.y_max - 1.5
        self.top_rect_y_max = self.y_max

        i = 0
        for k, v in self.instrs.items():
            #print(k,v)
            text = "{key:<6}: {val:>4.0f}".format(key=k, val=v)
            canvas.rect(self.top_rect_x_min,
                        self.top_rect_y_min - self.rect_y_offset * i,
                        self.top_rect_x_max,
                        self.top_rect_y_max - self.rect_y_offset * i)
            canvas.text(
                self.top_rect_x_min + self.text_x_offset, self.top_rect_y_min -
                self.rect_y_offset * i + self.text_y_offset, text)
            canvas.text(self.top_rect_x_min - 2,
                        self.top_rect_y_min - self.rect_y_offset * i,
                        "------------------>")
            i = i + 1


# Exec Block
@dataclass
class Exec(RectFrame):
    active_cus: float = 0.0
    vgprs: float = 0.0
    sgprs: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 8.0, self.y_max - 2.0, "Active CUs")
        canvas.text(self.x_min + 16.0,
                    self.y_max - 3.0,
                    str(self.active_cus),
                    color="yellow")

        canvas.rect(self.x_min + 2.0, self.y_min + 12.0, self.x_max - 2.0,
                    self.y_max - 4.0)
        canvas.text(self.x_min + 4.0, self.y_max - 6.0, r"VALU/MFMA")
        canvas.rect(self.x_min + 3.0, self.y_min + 15.0, self.x_max - 3.0,
                    self.y_max - 7.0)
        canvas.text(self.x_min + 4.0, self.y_max - 8.0,
                    "{key:<6}: {val:>3}".format(key="VGPRs", val=self.vgprs))

        canvas.rect(self.x_min + 2.0, self.y_min + 1.0, self.x_max - 2.0,
                    self.y_min + 9.0)
        canvas.text(self.x_min + 4.0, self.y_min + 7.0, r"SALU")
        canvas.rect(self.x_min + 3.0, self.y_min + 4.0, self.x_max - 3.0,
                    self.y_min + 6.0)
        canvas.text(self.x_min + 4.0, self.y_min + 5.0,
                    "{key:<6}: {val:>3}".format(key="SGPRs", val=self.sgprs))


# Wires between Exec block and GDS, LDS, Vector L1 cache
@dataclass
class WireEGLV(RectFrame):
    text_x_offset: float = 3.0

    gds_req: float = 0.0
    lds_idx: float = 0.0
    lds_dir: float = 0.0
    lds_interp: float = 0.0
    vector_L1_rd: float = 0.0
    vector_L1_wr: float = 0.0
    vector_L1_atomic: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 2.0,
                    "{key:<6}: {val:>4}".format(key="Req", val=self.gds_req))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 3.0,
                    "<-------------->")

        canvas.text(self.x_min + self.text_x_offset, self.y_max - 8.0,
                    "{key:<6}: {val:>4}".format(key="Idx", val=self.lds_idx))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 9.0,
                    "<-------------->")
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 10.0,
                    "{key:<6}: {val:>4}".format(key="Dir", val=self.lds_dir))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 11.0,
                    "<---------------")
        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 12.0,
            "{key:<6}: {val:>4}".format(key="Interp", val=self.lds_interp))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 13.0,
                    "--------------->")

        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 17.0,
            "{key:<6}: {val:>4}".format(key="Rd", val=self.vector_L1_rd))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 18.0,
                    "<---------------")
        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 19.0,
            "{key:<6}: {val:>4}".format(key="Wr", val=self.vector_L1_wr))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 20.0,
                    "--------------->")
        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 21.0,
            "{key:<6}: {val:>4}".format(key="Atomic",
                                        val=self.vector_L1_atomic))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 22.0,
                    "--------------->")


# GDS Block
@dataclass
class GDS(RectFrame):
    gws: float = 0.0
    latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_min + 2.5, self.x_max - 2.0,
                    self.y_max - 1.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 2.0,
            "{key:<4}: {val:>4} cycles".format(key="GWS", val=self.gws))

        canvas.rect(self.x_min + 2.0, self.y_min + 0.5, self.x_max - 2.0,
                    self.y_min + 2.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))


# LDS Block
@dataclass
class LDS(RectFrame):
    latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_min + 0.5, self.x_max - 2.0,
                    self.y_max - 3.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))


# Vector L1 Cache Block
@dataclass
class VectorL1Cache(RectFrame):
    ta_busy: float = 0.0
    td_busy: float = 0.0
    ta_stall: float = 0.0
    td_stall: float = 0.0
    tcp_hit: float = 0.0
    tcp_latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 8.5, self.y_max - 1.0,
                    "TA                              TCP")

        # TA
        canvas.rect(self.x_min + 2.0, self.y_max - 5.5, self.x_min + 17.0,
                    self.y_max - 1.5)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} %".format(key="Busy", val=self.ta_busy))

        # TD
        canvas.text(self.x_min + 8.5, self.y_max - 6.5, "TD")
        canvas.rect(self.x_min + 2.0, self.y_max - 11.5, self.x_min + 17.0,
                    self.y_max - 7.5)
        canvas.text(
            self.x_min + 4.0, self.y_max - 10.0,
            "{key:<4}: {val:>4} %".format(key="Busy", val=self.td_busy))

        # TA to TCP stall
        canvas.text(
            self.x_min + 18.0, self.y_max - 3.0,
            "{key:<5}: {val:>4} %".format(key="Stall", val=self.ta_stall))
        canvas.text(self.x_min + 18.0, self.y_max - 4.0, "------------->")

        # TCP to TD stall
        canvas.text(
            self.x_min + 18.0, self.y_max - 9.0,
            "{key:<5}: {val:>4} %".format(key="Stall", val=self.td_stall))
        canvas.text(self.x_min + 18.0, self.y_max - 10.0, "<-------------")

        # TCP
        canvas.rect(self.x_min + 32.0, self.y_max - 11.5, self.x_max - 2.0,
                    self.y_max - 1.5)
        canvas.text(self.x_min + 34.0, self.y_max - 9.0,
                    "{key:<4}: {val:>4} %".format(key="Hit", val=self.tcp_hit))
        canvas.text(
            self.x_min + 34.0, self.y_max - 10.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat",
                                               val=self.tcp_latency))


# Const L1 Cache
@dataclass
class ConstL1Cache(RectFrame):
    hit: float = 0.0
    latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_min + 2.5, self.x_max - 2.0,
                    self.y_max - 1.0)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0,
                    "{key:<4}: {val:>4} %".format(key="Hit", val=self.hit))

        canvas.rect(self.x_min + 2.0, self.y_min + 0.5, self.x_max - 2.0,
                    self.y_min + 2.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))


# Instr L1 Cache
@dataclass
class InstrL1Cache(RectFrame):
    hit: float = 0.0
    latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_min + 2.5, self.x_max - 2.0,
                    self.y_max - 1.0)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0,
                    "{key:<4}: {val:>4} %".format(key="Hit", val=self.hit))

        canvas.rect(self.x_min + 2.0, self.y_min + 0.5, self.x_max - 2.0,
                    self.y_min + 2.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))


# Wires between Vector L1 cache, Const L1 cache, Instr L1 cache and L2 Cache
@dataclass
class Wire_VCI_L2(RectFrame):
    text_v_x_offset: float = 30.0

    tcp_L2_rd: float = 0.0
    tcp_L2_wr: float = 0.0
    tcp_L2_atomic: float = 0.0
    constL1_L2_rd: float = 0.0
    constL1_L2_wr: float = 0.0
    constL1_L2_atomic: float = 0.0
    instrL1_L2_req: float = 0.0

    def draw(self, canvas):

        canvas.text(self.x_min + self.text_v_x_offset, self.y_max - 4.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.tcp_L2_rd))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 5.0,
                    "<---------------")
        canvas.text(self.x_min + self.text_v_x_offset, self.y_max - 6.0,
                    "{key:<6}: {val:>4}".format(key="Wr", val=self.tcp_L2_wr))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 7.0,
                    "--------------->")
        canvas.text(
            self.x_min + self.text_v_x_offset, self.y_max - 8.0,
            "{key:<6}: {val:>4}".format(key="Atomic", val=self.tcp_L2_atomic))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 9.0,
                    "--------------->")

        canvas.text(
            self.x_min, self.y_max - 16.0,
            "{key:<6}: {val:>4}".format(key="Rd", val=self.constL1_L2_rd))
        canvas.text(self.x_min - 2, self.y_max - 17.0, "<" + "-" * 45)
        canvas.text(
            self.x_min, self.y_max - 18.0,
            "{key:<6}: {val:>4}".format(key="Wr", val=self.constL1_L2_wr))
        canvas.text(self.x_min - 2, self.y_max - 19.0, "-" * 45 + ">")
        canvas.text(
            self.x_min, self.y_max - 20.0,
            "{key:<6}: {val:>4}".format(key="Atomic",
                                        val=self.constL1_L2_atomic))
        canvas.text(self.x_min - 2, self.y_max - 21.0, "-" * 45 + ">")

        canvas.text(
            self.x_min, self.y_max - 26.0,
            "{key:<6}: {val:>4}".format(key="Req", val=self.instrL1_L2_req))
        canvas.text(self.x_min - 2, self.y_max - 27.0, "<" + "-" * 45)


# L2 Cache
@dataclass
class L2Cache(RectFrame):
    rd: float = 0.0
    wr: float = 0.0
    atomic: float = 0.0
    hit: float = 0.0
    rd_cycles: float = 0.0
    wr_cycles: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_max - 24.0, self.x_max - 2.0,
                    self.y_max - 16.0)
        canvas.text(self.x_min + 3.0, self.y_max - 18.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.rd))
        canvas.text(self.x_min + 3.0, self.y_max - 20.0,
                    "{key:<6}: {val:>4}".format(key="Wr", val=self.wr))
        canvas.text(self.x_min + 3.0, self.y_max - 22.0,
                    "{key:<6}: {val:>4}".format(key="Atomic", val=self.atomic))

        canvas.rect(self.x_min + 2.0, self.y_max - 34.0, self.x_max - 2.0,
                    self.y_max - 26.0)
        canvas.text(self.x_min + 3.0, self.y_max - 28.0,
                    "{key:<6}: {val:>4} %".format(key="Hit", val=self.hit))
        canvas.text(
            self.x_min + 3.0, self.y_max - 30.0,
            "{key:<6}: {val:>4} cycles".format(key="Rd", val=self.rd_cycles))
        canvas.text(
            self.x_min + 3.0, self.y_max - 32.0,
            "{key:<6}: {val:>4} cycles".format(key="Wr", val=self.wr_cycles))


# Wires between L2 block and EA
@dataclass
class Wire_L2_EA(RectFrame):
    text_x_offset: float = 3.0

    rd: float = 0.0
    wr: float = 0.0
    atomic: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 2.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.rd))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 3.0,
                    "<---------------")
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 4.0,
                    "{key:<6}: {val:>4}".format(key="Wr", val=self.wr))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 5.0,
                    "--------------->")
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 6.0,
                    "{key:<6}: {val:>4}".format(key="Atomic", val=self.atomic))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 7.0,
                    "--------------->")


def plot_mem_chat(df, arch):
    '''plot memory chat from given df and arch'''

    mem_chart_data = {}
    mem_chart_data["wave_occupancy"] = 28
    mem_chart_data["wave_life"] = 3185

    mem_chart_data["instr_scalar"] = 20
    mem_chart_data["instr_valu"] = 30
    mem_chart_data["instr_vmem"] = 1118
    mem_chart_data["instr_mfma"] = 2
    mem_chart_data["instr_lds"] = 11
    mem_chart_data["instr_gds"] = 0
    mem_chart_data["instr_flat"] = 6
    mem_chart_data["instr_branch"] = 4

    mem_chart_data["active_cus"] = 58
    mem_chart_data["vgprs"] = 8
    mem_chart_data["sgprs"] = 24

    mem_chart_data["gds_req"] = 0.0
    mem_chart_data["lds_idx"] = 1.0
    mem_chart_data["lds_dir"] = 0.0
    mem_chart_data["lds_interp"] = 0.0
    mem_chart_data["vector_L1_rd"] = 16.0
    mem_chart_data["vector_L1_wr"] = 17.0
    mem_chart_data["vector_L1_atomic"] = 0.0

    mem_chart_data["gds_gws"] = 0.0
    mem_chart_data["gds_latency"] = 0.0

    mem_chart_data["lds_latency"] = 40.0

    mem_chart_data["ta_busy"] = 30.0
    mem_chart_data["td_busy"] = 40.0
    mem_chart_data["ta_stall"] = 1.0
    mem_chart_data["td_stall"] = 2.0
    mem_chart_data["tcp_hit"] = 50.0
    mem_chart_data["tcp_latency"] = 879

    mem_chart_data["const_L1_hit"] = 100.0
    mem_chart_data["const_L1_latency"] = 192

    mem_chart_data["instr_L1_hit"] = 100.0
    mem_chart_data["instr_L1_latency"] = 9.0

    mem_chart_data["tcp_L2_rd"] = 2
    mem_chart_data["tcp_L2_wr"] = 0
    mem_chart_data["tcp_L2_atomic"] = 0
    mem_chart_data["constL1_L2_rd"] = 2
    mem_chart_data["constL1_L2_wr"] = 1
    mem_chart_data["constL1_L2_atomic"] = 0
    mem_chart_data["instrL1_L2_req"] = 3

    mem_chart_data["L2_rd"] = 2
    mem_chart_data["L2_wr"] = 0
    mem_chart_data["L2_atomic"] = 0
    mem_chart_data["L2_hit"] = 2
    mem_chart_data["L2_rd_cycles"] = 601
    mem_chart_data["L2_wr_cycles"] = 3450

    mem_chart_data["L2_EA_rd"] = 3
    mem_chart_data["L2_EA_wr"] = 1
    mem_chart_data["L2_EA_atomic"] = 0

    # Memory chart top pannel for 1 instance
    class MemChart:

        def __init__(self, x_min, y_min, x_max, y_max):
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

        def draw(self, canvas, mem_chart_data):

            #######################################################################
            # Overall rect and title
            canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
            canvas.text(self.x_min + 2.0, self.y_max - 2.0,
                        "Memory Chart(Normalization: per Sec)")

            #######################################################################
            # Instr Buff Block
            block_instr_buff = InstrBuff(label="Instr Buff")
            block_instr_buff.x_min = 2.0
            block_instr_buff.x_max = block_instr_buff.x_min + 27.0
            block_instr_buff.y_max = self.y_max - 5.0
            block_instr_buff.y_min = block_instr_buff.y_max - 24.0

            block_instr_buff.wave_occupancy = mem_chart_data["wave_occupancy"]
            block_instr_buff.wave_life = mem_chart_data["wave_life"]

            block_instr_buff.draw(canvas)

            #######################################################################
            # Instr Dispatch Block
            block_instr_disp = InstrDispatch(label="Instr Dispatch")
            block_instr_disp.x_min = block_instr_buff.x_max + 9.0
            block_instr_disp.x_max = block_instr_disp.x_min + 20.0
            block_instr_disp.y_max = block_instr_buff.y_max
            block_instr_disp.y_min = block_instr_buff.y_min

            block_instr_disp.instrs["SCALAR"] = mem_chart_data["instr_scalar"]
            block_instr_disp.instrs["VALU"] = mem_chart_data["instr_valu"]
            block_instr_disp.instrs["VMEM"] = mem_chart_data["instr_vmem"]
            block_instr_disp.instrs["MFMA"] = mem_chart_data["instr_mfma"]
            block_instr_disp.instrs["LDS"] = mem_chart_data["instr_lds"]
            block_instr_disp.instrs["GDS"] = mem_chart_data["instr_gds"]
            block_instr_disp.instrs["FLAT"] = mem_chart_data["instr_flat"]
            block_instr_disp.instrs["BRANCH"] = mem_chart_data["instr_branch"]

            block_instr_disp.draw(canvas)

            #######################################################################
            # Exec Block
            block_exec = Exec(label="Exec")
            block_exec.x_min = block_instr_disp.x_max
            block_exec.x_max = block_exec.x_min + 20
            block_exec.y_min = block_instr_disp.y_min
            block_exec.y_max = block_instr_disp.y_max

            block_exec.active_cus = mem_chart_data["active_cus"]
            block_exec.vgprs = mem_chart_data["vgprs"]
            block_exec.sgprs = mem_chart_data["sgprs"]

            block_exec.draw(canvas)

            #######################################################################
            # Wires between Exec block and GDS, LDS, Vector L1 cache
            wires_EGLV = WireEGLV(label="WireEGLV")
            wires_EGLV.x_min = block_exec.x_max
            wires_EGLV.x_max = wires_EGLV.x_min + 16
            wires_EGLV.y_min = block_instr_disp.y_min
            wires_EGLV.y_max = block_instr_disp.y_max

            wires_EGLV.active_cus = mem_chart_data["active_cus"]
            wires_EGLV.vgprs = mem_chart_data["vgprs"]
            wires_EGLV.sgprs = mem_chart_data["sgprs"]

            wires_EGLV.gds_req = mem_chart_data["gds_req"]
            wires_EGLV.lds_idx = mem_chart_data["lds_idx"]
            wires_EGLV.lds_dir = mem_chart_data["lds_dir"]
            wires_EGLV.lds_interp = mem_chart_data["lds_interp"]
            wires_EGLV.vector_L1_rd = mem_chart_data["vector_L1_rd"]
            wires_EGLV.vector_L1_wr = mem_chart_data["vector_L1_wr"]
            wires_EGLV.vector_L1_atomic = mem_chart_data["vector_L1_atomic"]

            wires_EGLV.draw(canvas)

            #######################################################################
            # GDS block
            block_gds = GDS(label="GDS")
            block_gds.x_min = wires_EGLV.x_max + 1
            block_gds.x_max = block_gds.x_min + 24
            block_gds.y_max = wires_EGLV.y_max
            block_gds.y_min = block_gds.y_max - 5

            block_gds.gws = mem_chart_data["gds_gws"]
            block_gds.latency = mem_chart_data["gds_latency"]

            block_gds.draw(canvas)

            #######################################################################
            # LDS block
            block_lds = LDS(label="LDS")
            block_lds.x_min = block_gds.x_min
            block_lds.x_max = block_gds.x_max
            block_lds.y_max = block_gds.y_min - 3
            block_lds.y_min = block_lds.y_max - 5

            block_lds.latency = mem_chart_data["lds_latency"]

            block_lds.draw(canvas)

            #######################################################################
            # Vector L1 Cache Block
            block_vector_L1 = VectorL1Cache(label="Vector L1 Cache")
            block_vector_L1.x_min = block_lds.x_min
            block_vector_L1.x_max = block_vector_L1.x_min + 54
            block_vector_L1.y_max = block_lds.y_min - 3
            block_vector_L1.y_min = block_vector_L1.y_max - 13

            block_vector_L1.ta_busy = mem_chart_data["ta_busy"] = 30.0
            block_vector_L1.td_busy = mem_chart_data["td_busy"] = 40.0
            block_vector_L1.ta_stall = mem_chart_data["ta_stall"] = 1.0
            block_vector_L1.td_stall = mem_chart_data["td_stall"] = 2.0
            block_vector_L1.tcp_hit = mem_chart_data["tcp_hit"] = 50.0
            block_vector_L1.tcp_latency = mem_chart_data["tcp_latency"] = 879

            block_vector_L1.draw(canvas)

            #######################################################################
            # Const L1 Cache block
            block_const_L1 = ConstL1Cache(label="Const L1 Cache")
            block_const_L1.x_min = block_lds.x_min
            block_const_L1.x_max = block_lds.x_max
            block_const_L1.y_max = block_vector_L1.y_min - 3
            block_const_L1.y_min = block_const_L1.y_max - 5

            block_const_L1.hit = mem_chart_data["const_L1_hit"]
            block_const_L1.latency = mem_chart_data["const_L1_latency"]

            block_const_L1.draw(canvas)

            #######################################################################
            # Instr L1 Cache block
            block_instr_L1 = InstrL1Cache(label="Instr L1 Cache")
            block_instr_L1.x_min = block_const_L1.x_min
            block_instr_L1.x_max = block_const_L1.x_max
            block_instr_L1.y_max = block_const_L1.y_min - 3
            block_instr_L1.y_min = block_instr_L1.y_max - 5

            block_instr_L1.hit = mem_chart_data["instr_L1_hit"]
            block_instr_L1.latency = mem_chart_data["instr_L1_latency"]

            block_instr_L1.draw(canvas)

            #######################################################################
            # Wires between Vector L1 cache, Const L1 cache, Instr L1 cache and L2 Cache
            wires_VCI_L2 = Wire_VCI_L2(label="Wire_VCI_L2")
            wires_VCI_L2.x_min = block_instr_L1.x_max + 4
            wires_VCI_L2.x_max = wires_VCI_L2.x_min + 44
            wires_VCI_L2.y_min = block_instr_L1.y_min
            wires_VCI_L2.y_max = block_vector_L1.y_max
            wires_VCI_L2.tcp_L2_rd = mem_chart_data["tcp_L2_rd"]
            wires_VCI_L2.tcp_L2_wr = mem_chart_data["tcp_L2_wr"]
            wires_VCI_L2.tcp_L2_atomic = mem_chart_data["tcp_L2_atomic"]
            wires_VCI_L2.constL1_L2_rd = mem_chart_data["constL1_L2_rd"]
            wires_VCI_L2.constL1_L2_wr = mem_chart_data["constL1_L2_wr"]
            wires_VCI_L2.constL1_L2_atomic = mem_chart_data["constL1_L2_atomic"]
            wires_VCI_L2.instrL1_L2_req = mem_chart_data["instrL1_L2_req"]

            wires_VCI_L2.draw(canvas)

            #######################################################################
            # L2 Cache block
            block_L2 = L2Cache(label="L2 Cache")

            block_L2.x_min = wires_VCI_L2.x_max + 1
            block_L2.x_max = block_L2.x_min + 24
            block_L2.y_min = block_instr_L1.y_min
            block_L2.y_max = block_lds.y_max

            block_L2.rd = mem_chart_data["L2_rd"]
            block_L2.wr = mem_chart_data["L2_wr"]
            block_L2.atomic = mem_chart_data["L2_atomic"]
            block_L2.hit = mem_chart_data["L2_hit"]
            block_L2.rd_cycles = mem_chart_data["L2_rd_cycles"]
            block_L2.wr_cycles = mem_chart_data["L2_wr_cycles"]

            block_L2.draw(canvas)

            #######################################################################
            # Wires between L2 block and EA
            wires_L2_EA = Wire_L2_EA(label="Wire_L2_EA",
                                    x_min=block_L2.x_max,
                                    x_max=self.x_min + 16,
                                    y_min=block_instr_L1.y_min + 10,
                                    y_max=block_instr_L1.y_min + 20)

            wires_L2_EA.rd = mem_chart_data["L2_EA_rd"]
            wires_L2_EA.wr = mem_chart_data["L2_EA_wr"]
            wires_L2_EA.atomic = mem_chart_data["L2_EA_atomic"]

            wires_L2_EA.draw(canvas)


    canvas = Canvas(width=260, height=54, xmax=260, ymax=54)

    mc = MemChart(0, 0, 259, 53)
    mc.draw(canvas, mem_chart_data)

    # return the plot string stream
    return canvas.plot()


if __name__ == '__main__':
    df = ''
    arch = ''
    print(plot_mem_chat(df, arch))

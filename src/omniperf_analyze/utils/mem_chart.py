###############################################################################
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
###############################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Generator
from types import SimpleNamespace as NS
from plotille import Canvas


# A basic rect frame for any block or group of wires where all its elements should
# be within this range, except: (a) the label(title) might be on the top of it,
# (b) some wires around it don't have to be grouped specifically.

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


# Wires between Instr Buff and Instr Dispatch
@dataclass
class Wire_InstrBuff_InstrDispatch(RectFrame):
    def draw(self, canvas):
        # Todo: finer wires for connections
        canvas.line(self.x_min + 2, self.y_min, self.x_min + 2, self.y_max)
        canvas.line(self.x_max, self.y_min + 1.5, self.x_max, self.y_max - 1.5)
        canvas.line(self.x_min + 2, self.y_min, self.x_max, self.y_min + 1.5)
        canvas.line(self.x_min + 2, self.y_max, self.x_max - 0.5, self.y_max - 1.5)


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
    active_cus: int = 0
    num_cus: int = 0
    vgprs: int = 0
    sgprs: int = 0
    lds_alloc: int = 0
    scratch_alloc: int = 0
    wavefronts: int = 0
    workgroups: int = 0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)

        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 2.0, self.y_max - 2.0, "Active CUs")
        canvas.text(self.x_min + 2.0,
                    self.y_max - 3.0,
                    "{a}/{n}".format(a=self.active_cus, n=self.num_cus),
                    color="yellow")

        canvas.rect(self.x_min + 2.0, self.y_max - 7.0, self.x_max - 2.0,
                    self.y_max - 5.0)
        canvas.text(self.x_min + 4.0, self.y_max - 6.0,
                    "{key:<6}: {val:>5}".format(key="VGPRs", val=self.vgprs))

        canvas.rect(self.x_min + 2.0, self.y_max - 10.0, self.x_max - 2.0,
                    self.y_max - 8.0)
        canvas.text(self.x_min + 4.0, self.y_max - 9.0,
                    "{key:<6}: {val:>5}".format(key="SGPRs", val=self.sgprs))

        canvas.rect(self.x_min + 2.0, self.y_max - 15.0, self.x_max - 2.0,
                    self.y_max - 12.0)
        canvas.text(self.x_min + 4.0, self.y_max - 13.0, "LDS Alloc:")
        canvas.text(self.x_min + 4.0, self.y_max - 14.0, "{val:>13}".format(val=self.lds_alloc))

        canvas.rect(self.x_min + 2.0, self.y_max - 19.0, self.x_max - 2.0,
                    self.y_max - 16.0)
        canvas.text(self.x_min + 4.0, self.y_max - 17.0, "Scratch Alloc:") 
        canvas.text(self.x_min + 4.0, self.y_max - 18.0, "{val:>13}".format(val=self.scratch_alloc))

        canvas.rect(self.x_min + 2.0, self.y_max - 24.0, self.x_max - 2.0,
                    self.y_max - 21.0)
        canvas.text(self.x_min + 4.0, self.y_max - 22.0, "Wavefronts:")
        canvas.text(self.x_min + 4.0, self.y_max - 23.0, "{val:>13}".format(val=self.wavefronts))

        canvas.rect(self.x_min + 2.0, self.y_max - 28.0, self.x_max - 2.0,
                    self.y_max - 25.0)
        canvas.text(self.x_min + 4.0, self.y_max - 26.0, "Workgroups:") 
        canvas.text(self.x_min + 4.0, self.y_max - 27.0, "{val:>13}".format(val=self.workgroups))


# Wires between Exec block and GDS, LDS, Vector L1 cache, Scalar L1D Cache
@dataclass
class Wire_E_GLVS(RectFrame):
    text_x_offset: float = 3.0

    lds_req: float = 0.0
    vector_L1_rd: float = 0.0
    vector_L1_wr: float = 0.0
    vector_L1_atomic: float = 0.0
    scalar_L1D_rd: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 2.0,
                    "{key:<6}: {val:>4}".format(key="Req", val=self.lds_req))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 3.0,
                    "<---------------")

        canvas.text(self.x_min + self.text_x_offset, self.y_max - 10.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.vector_L1_rd))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 11.0,
                    "<---------------")
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 12.0,
                    "{key:<6}: {val:>4}".format(key="Wt", val=self.vector_L1_wr))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 13.0,
                    "--------------->")
        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 14.0,
            "{key:<6}: {val:>4}".format(key="Atomic", val=self.vector_L1_atomic))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 15.0,
                    "<-------------->")

        canvas.text(
            self.x_min + self.text_x_offset, self.y_max - 22.0,
            "{key:<6}: {val:>4}".format(key="Rd", val=self.scalar_L1D_rd))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 23.0,
                    "<---------------")


# Wire between Instr Buff and Instr L1 Cache
@dataclass
class Wire_InstrBuff_IL1Cache(RectFrame):
    fetch: int = 0

    def draw(self, canvas):
        end_col = int(self.y_max - self.y_min)
        canvas.text(self.x_min, self.y_max - 1, "^")
        for i in range(2, end_col):
            canvas.text(self.x_min, self.y_max - i, "|")
        canvas.text(self.x_min + 27, self.y_max - end_col + 1,
                    "{key:<6}: {val:>4}".format(key="Fetch", val=self.fetch))
        canvas.text(self.x_min, self.y_max - end_col,
                    "-" * (int(self.x_max - self.x_min)))


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
    util: float = 0.0
    latency: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_max - 2.5, self.x_max - 2.0,
                    self.y_max - 1.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 2.0,
            "{key:<4}: {val:>4} %".format(key="Util", val=self.util))
        
        canvas.rect(self.x_min + 2.0, self.y_max - 4.5, self.x_max - 2.0,
                    self.y_max - 3.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))


# Vector L1 Cache Block
@dataclass
class VectorL1Cache(RectFrame):
    hit: float = 0.0
    latency: float = 0.0
    coales: float = 0.0
    stall: float = 0.0

    def draw(self, canvas):
        canvas.text(self.x_min, self.y_max + 1.0, self.label)
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)

        canvas.rect(self.x_min + 2.0, self.y_max - 2.5, self.x_max - 2.0,
                    self.y_max - 1.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 2.0,
            "{key:<4}: {val:>4} %".format(key="hit", val=self.hit))
        
        canvas.rect(self.x_min + 2.0, self.y_max - 4.5, self.x_max - 2.0,
                    self.y_max - 3.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 4.0,
            "{key:<4}: {val:>4} cycles".format(key="Lat", val=self.latency))
        

        canvas.rect(self.x_min + 2.0, self.y_max - 6.5, self.x_max - 2.0,
                    self.y_max - 5.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 6.0,
            "{key:<4}: {val:>4} %".format(key="Coales", val=self.coales))
        
        canvas.rect(self.x_min + 2.0, self.y_max - 8.5, self.x_max - 2.0,
                    self.y_max - 7.0)
        canvas.text(
            self.x_min + 4.0, self.y_max - 8.0,
            "{key:<4}: {val:>4} cycles".format(key="Stall", val=self.stall))
            

# Scalar L1D Cache
@dataclass
class ScalarL1DCache(RectFrame):
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


# Wires between Vector L1 cache, Scalar L1D Cache, Instr L1 cache and L2 Cache
@dataclass
class Wire_VSI_L2(RectFrame):
    text_v_x_offset: float = 0.0

    vl1_l2_rd: float = 0.0
    vl1_l2_wr: float = 0.0
    vl1_l2_atomic: float = 0.0
    constL1_L2_rd: float = 0.0
    constL1_L2_wr: float = 0.0
    constL1_L2_atomic: float = 0.0
    instrL1_L2_req: float = 0.0

    def draw(self, canvas):

        canvas.text(self.x_min + self.text_v_x_offset, self.y_max - 2.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.vl1_l2_rd))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 3.0,
                    "<---------------")
        canvas.text(self.x_min + self.text_v_x_offset, self.y_max - 4.0,
                    "{key:<6}: {val:>4}".format(key="Wr", val=self.vl1_l2_wr))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 5.0,
                    "--------------->")
        canvas.text(
            self.x_min + self.text_v_x_offset, self.y_max - 6.0,
            "{key:<6}: {val:>4}".format(key="Atomic", val=self.vl1_l2_atomic))
        canvas.text(self.x_min + self.text_v_x_offset - 2, self.y_max - 7.0,
                    "<-------------->")

        canvas.text(
            self.x_min, self.y_max - 12.0,
            "{key:<6}: {val:>4}".format(key="Rd", val=self.constL1_L2_rd))
        canvas.text(self.x_min - 2, self.y_max - 13.0, "<---------------")
        canvas.text(
            self.x_min, self.y_max - 14.0,
            "{key:<6}: {val:>4}".format(key="Wr", val=self.constL1_L2_wr))
        canvas.text(self.x_min - 2, self.y_max - 15.0, "--------------->")
        canvas.text(
            self.x_min, self.y_max - 16.0,
            "{key:<6}: {val:>4}".format(key="Atomic",
                                        val=self.constL1_L2_atomic))
        canvas.text(self.x_min - 2, self.y_max - 17.0, "<-------------->")

        canvas.text(
            self.x_min, self.y_max - 22.0,
            "{key:<6}: {val:>4}".format(key="Req", val=self.instrL1_L2_req))
        canvas.text(self.x_min - 2, self.y_max - 23.0, "<---------------")


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

        canvas.rect(self.x_min + 2.0, self.y_max - 5.0, self.x_max - 2.0,
                    self.y_max - 3.0)
        canvas.text(self.x_min + 4.0, self.y_max - 4.0,
                    "{key:<6}: {val:>4} %".format(key="Hit", val=self.hit))

        canvas.text(self.x_min + 2.0, self.y_max - 7.0, "Request")
        canvas.rect(self.x_min + 2.0, self.y_max - 16.0, self.x_max - 2.0,
                    self.y_max - 7.5)
        canvas.text(self.x_min + 4.0, self.y_max - 10.0,
                    "{key:<6}: {val:>4}".format(key="Rd", val=self.rd))
        canvas.text(self.x_min + 4.0, self.y_max - 12.0,
                    "{key:<6}: {val:>4}".format(key="Wr", val=self.wr))
        canvas.text(self.x_min + 4.0, self.y_max - 14.0,
                    "{key:<6}: {val:>4}".format(key="Atomic", val=self.atomic))

        canvas.text(self.x_min + 2.0, self.y_max - 19.0, "Latency (cycles)")
        canvas.rect(self.x_min + 2.0, self.y_max - 25.0, self.x_max - 2.0,
                    self.y_max - 19.5)

        canvas.text(
            self.x_min + 4.0, self.y_max - 22.0,
            "{key:<6}: {val:>4}".format(key="Rd", val=self.rd_cycles))
        canvas.text(
            self.x_min + 4.0, self.y_max - 24.0,
            "{key:<6}: {val:>4}".format(key="Wr", val=self.wr_cycles))


# Wires between L2 block and Fabric
@dataclass
class Wire_L2_Fabric(RectFrame):
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


# xGMI/PCIe block with wires to fabric
@dataclass
class xGMI_PCIe(RectFrame):
    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 1.0, self.y_max - 2.0, self.label)
        canvas.text(self.x_min + 3.0 , self.y_max - 5.0, "^   |")
        canvas.text(self.x_min + 3.0 , self.y_max - 6.0, "|   |")
        canvas.text(self.x_min + 3.0 , self.y_max - 7.0, "|   |")
        canvas.text(self.x_min + 3.0 , self.y_max - 8.0, "|   v")


# Fabric Cache Block
@dataclass
class Fabric(RectFrame):
    rd: int = 0
    wr: int = 0
    atomic: int = 0

    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 6.0, self.y_max - 2.0, "   " + self.label)
        canvas.text(self.x_min + 2.0, self.y_max - 4.0, "Latency (cycles)")

        canvas.rect(self.x_min + 2.0, self.y_max - 9, self.x_max - 2.0,
                    self.y_max - 4.5)
        canvas.text(self.x_min + 4.0, self.y_max - 5.5,
                    "{key:<6}: {val:>6}".format(key="Rd", val=self.rd))
        canvas.text(self.x_min + 4.0, self.y_max - 6.5,
                    "{key:<6}: {val:>6} ".format(key="Wr", val=self.wr))
        canvas.text(self.x_min + 4.0, self.y_max - 7.5,
                    "{key:<6}: {val:>6}".format(key="Atomic", val=self.atomic))


# GMI block with wires to fabric
@dataclass
class GMI(RectFrame):
    def draw(self, canvas):
        canvas.text(self.x_min + 3.0 , self.y_max + 4.0, "^   |")
        canvas.text(self.x_min + 3.0 , self.y_max + 3.0, "|   |")
        canvas.text(self.x_min + 3.0 , self.y_max + 2.0, "|   |")
        canvas.text(self.x_min + 3.0 , self.y_max + 1.0, "|   v")
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0, self.label)


# Wires between fabric and HBM
@dataclass
class Wire_Fabric_HBM(RectFrame):
    text_x_offset: float = 3.0

    rd: int = 0
    wr: int = 0

    def draw(self, canvas):
        canvas.text(self.x_min + self.text_x_offset, self.y_max,
                    "{key:<2}: {val:>4}".format(key="Rd", val=self.rd))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 1.0,
                    "<-----------")
        canvas.text(self.x_min + self.text_x_offset, self.y_max - 2.0,
                    "{key:<2}: {val:>4}".format(key="Wr", val=self.wr))
        canvas.text(self.x_min + self.text_x_offset - 2, self.y_max - 3.0,
                    "----------->")


# HBM
@dataclass
class HBM(RectFrame):
    def draw(self, canvas):
        canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
        canvas.text(self.x_min + 4.0, self.y_max - 2.0, self.label)


def plot_mem_chart(df, arch):
    '''plot memory chart from given df and arch'''

    mem_chart_data = {}
    mem_chart_data["wave_occupancy"] = 28 #fixme
    mem_chart_data["Wave Life"] = 3185

    mem_chart_data["SALU"] = 20
    mem_chart_data["SMEM"] = 6
    mem_chart_data["VALU"] = 30
    mem_chart_data["MFMA"] = 2
    mem_chart_data["VMEM"] = 1118
    mem_chart_data["LDS"] = 11
    mem_chart_data["GWS"] = 0
    mem_chart_data["BR"] = 4

    mem_chart_data["Active CUs"] = 58
    mem_chart_data["Num CUs"] = 120
    mem_chart_data["VGPR"] = 8
    mem_chart_data["SGPR"] = 24
    mem_chart_data["LDS Allocation"] = 24000
    mem_chart_data["Scratch Allocation"] = 879
    mem_chart_data["Wavefronts"] = 16
    mem_chart_data["Workgroups"] = 8000

    mem_chart_data["LDS Req"] = 0.0
    mem_chart_data["VL1 Rd"] = 16.0
    mem_chart_data["VL1 Wr"] = 17.0
    mem_chart_data["VL1 Atomic"] = 0.0
    mem_chart_data["scalar_L1D_rd"] = 1.0 #fixme

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

    mem_chart_data["VL1_L2 Rd"] = 2
    mem_chart_data["VL1_L2 Wr"] = 0
    mem_chart_data["VL1_L2 Atomic"] = 0
    mem_chart_data["constL1_L2_rd"] = 2
    mem_chart_data["constL1_L2_wr"] = 1
    mem_chart_data["constL1_L2_atomic"] = 0
    mem_chart_data["instrL1_L2_req"] = 3

    mem_chart_data["L2 Rd"] = 2
    mem_chart_data["L2 Wr"] = 0
    mem_chart_data["L2 Atomic"] = 0
    mem_chart_data["L2 Hit"] = 2
    mem_chart_data["L2 Rd Lat"] = 601
    mem_chart_data["L2 Wr Lat"] = 3450

    mem_chart_data["L2_Fabric_rd"] = 3
    mem_chart_data["L2_Fabric_wr"] = 1
    mem_chart_data["L2_Fabric_atomic"] = 0

    # Memory chart top pannel for 1 instance
    class MemChart:

        def __init__(self, x_min, y_min, x_max, y_max):
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

        def draw(self, canvas, mem_chart_data):

            # ----------------------------------------
            # Overall rect and title
            canvas.rect(self.x_min, self.y_min, self.x_max, self.y_max)
            canvas.text(self.x_min + 2.0, self.y_max - 2.0,
                        "Memory Chart(Normalization: per Sec)")

            # ----------------------------------------
            # Instr Buff Block
            block_instr_buff = InstrBuff(label="Instr Buff")
            block_instr_buff.x_min = 2.0
            block_instr_buff.x_max = block_instr_buff.x_min + 27.0
            block_instr_buff.y_max = self.y_max - 5.0
            block_instr_buff.y_min = block_instr_buff.y_max - 24.0

            block_instr_buff.wave_occupancy = mem_chart_data["wave_occupancy"]
            block_instr_buff.wave_life = mem_chart_data["Wave Life"]

            block_instr_buff.draw(canvas)

            # ----------------------------------------
            # Wires between Instr Buff and Instr Dispatch
            wire_I_I = Wire_InstrBuff_InstrDispatch(label="Wire_InstrBuff_InstrDispatch",
                                    x_min=block_instr_buff.x_max + 1,
                                    x_max=block_instr_buff.x_max + 7,
                                    y_min=block_instr_buff.y_min,
                                    y_max=block_instr_buff.y_max)
            wire_I_I.draw(canvas)

            # ----------------------------------------
            # Instr Dispatch Block
            block_instr_disp = InstrDispatch(label="Instr Dispatch")
            block_instr_disp.x_min = block_instr_buff.x_max + 9.0
            block_instr_disp.x_max = block_instr_disp.x_min + 20.0
            block_instr_disp.y_max = block_instr_buff.y_max
            block_instr_disp.y_min = block_instr_buff.y_min

            block_instr_disp.instrs["SALU"] = mem_chart_data["SALU"]
            block_instr_disp.instrs["SMEM"] = mem_chart_data["SMEM"]
            block_instr_disp.instrs["VALU"] = mem_chart_data["VALU"]
            block_instr_disp.instrs["MFMA"] = mem_chart_data["MFMA"]
            block_instr_disp.instrs["VMEM"] = mem_chart_data["VMEM"]
            block_instr_disp.instrs["LDS"] = mem_chart_data["LDS"]
            block_instr_disp.instrs["GWS"] = mem_chart_data["GWS"]
            block_instr_disp.instrs["BRANCH"] = mem_chart_data["BR"]

            block_instr_disp.draw(canvas)

            # ----------------------------------------
            # Exec Block
            block_exec = Exec(label="Exec")
            block_exec.x_min = block_instr_disp.x_max
            block_exec.x_max = block_exec.x_min + 20
            block_exec.y_min = block_instr_disp.y_min - 6
            block_exec.y_max = block_instr_disp.y_max

            block_exec.active_cus = mem_chart_data["Active CUs"]
            block_exec.num_cus = mem_chart_data["Num CUs"]
            block_exec.vgprs = mem_chart_data["VGPR"]
            block_exec.sgprs = mem_chart_data["SGPR"]
            block_exec.lds_alloc = mem_chart_data["LDS Allocation"]
            block_exec.scratch_alloc = mem_chart_data["Scratch Allocation"]
            block_exec.wavefronts = mem_chart_data["Wavefronts"]
            block_exec.workgroups = mem_chart_data["Workgroups"]

            block_exec.draw(canvas)

            # ----------------------------------------
            # Wires between Exec block and GDS, LDS, Vector L1 cache
            wires_EGLV = Wire_E_GLVS(label="Wire_E_GLVS")
            wires_EGLV.x_min = block_exec.x_max
            wires_EGLV.x_max = wires_EGLV.x_min + 16
            wires_EGLV.y_min = block_instr_disp.y_min
            wires_EGLV.y_max = block_instr_disp.y_max

            # fixme
            wires_EGLV.lds_req = mem_chart_data["LDS Req"]
            wires_EGLV.vector_L1_rd = mem_chart_data["VL1 Rd"]
            wires_EGLV.vector_L1_wr = mem_chart_data["VL1 Wr"]
            wires_EGLV.vector_L1_atomic = mem_chart_data["VL1 Atomic"]
            wires_EGLV.scalar_L1D_rd = mem_chart_data["scalar_L1D_rd"]

            wires_EGLV.draw(canvas)

            # ----------------------------------------
            # Wire between Instr Buff and Instr L1 Cache
            wire_InstrBuff_IL1Cache = Wire_InstrBuff_IL1Cache(label="Wire_InstrBuff_IL1Cache",
                                                              x_min=block_instr_buff.x_max / 2,
                                                              x_max=block_instr_buff.x_max / 2 + 80,
                                                              y_min=block_exec.y_min - 1,
                                                              y_max=block_instr_buff.y_min)
            
            # fixme wire_InstrBuff_IL1Cache.fetch = 

            wire_InstrBuff_IL1Cache.draw(canvas)

            # ----------------------------------------
            # GDS block
            # block_gds = GDS(label="GDS")
            # block_gds.x_min = wires_EGLV.x_max + 1
            # block_gds.x_max = block_gds.x_min + 24
            # block_gds.y_max = wires_EGLV.y_max
            # block_gds.y_min = block_gds.y_max - 5

            # block_gds.gws = mem_chart_data["gds_gws"]
            # block_gds.latency = mem_chart_data["gds_latency"]

            # block_gds.draw(canvas)

            # ----------------------------------------
            # LDS block
            block_lds = LDS(label="LDS")
            block_lds.x_min = wires_EGLV.x_max + 1
            block_lds.x_max = block_lds.x_min + 24
            block_lds.y_max = wires_EGLV.y_max
            block_lds.y_min = block_lds.y_max - 5

            #fixme
            block_lds.latency = mem_chart_data["lds_latency"]

            block_lds.draw(canvas)

            # ----------------------------------------
            # Vector L1 Cache Block
            block_vector_L1 = VectorL1Cache(label="Vector L1 Cache")
            block_vector_L1.x_min = block_lds.x_min
            block_vector_L1.x_max = block_lds.x_max
            block_vector_L1.y_max = block_lds.y_min - 3
            block_vector_L1.y_min = block_vector_L1.y_max - 9

            #fixme
            # block_vector_L1.ta_busy = mem_chart_data["ta_busy"] = 30.0
            # block_vector_L1.td_busy = mem_chart_data["td_busy"] = 40.0
            # block_vector_L1.ta_stall = mem_chart_data["ta_stall"] = 1.0
            # block_vector_L1.td_stall = mem_chart_data["td_stall"] = 2.0
            # block_vector_L1.tcp_hit = mem_chart_data["tcp_hit"] = 50.0
            # block_vector_L1.tcp_latency = mem_chart_data["tcp_latency"] = 879

            block_vector_L1.draw(canvas)

            # ----------------------------------------
            # Scalar L1D Cache block
            block_const_L1 = ScalarL1DCache(label="Scalar L1D Cache")
            block_const_L1.x_min = block_lds.x_min
            block_const_L1.x_max = block_lds.x_max
            block_const_L1.y_max = block_vector_L1.y_min - 3
            block_const_L1.y_min = block_const_L1.y_max - 5

            block_const_L1.hit = mem_chart_data["const_L1_hit"]
            block_const_L1.latency = mem_chart_data["const_L1_latency"]

            block_const_L1.draw(canvas)

            # ----------------------------------------
            # Instr L1 Cache block
            block_instr_L1 = InstrL1Cache(label="Instr L1 Cache")
            block_instr_L1.x_min = block_const_L1.x_min
            block_instr_L1.x_max = block_const_L1.x_max
            block_instr_L1.y_max = block_const_L1.y_min - 3
            block_instr_L1.y_min = block_instr_L1.y_max - 5

            block_instr_L1.hit = mem_chart_data["instr_L1_hit"]
            block_instr_L1.latency = mem_chart_data["instr_L1_latency"]

            block_instr_L1.draw(canvas)

            # ----------------------------------------
            # Wires between Vector L1 cache, Const L1 cache, Instr L1 cache and L2 Cache
            wires_VSI_L2Rd = Wire_VSI_L2(label="Wire_VSI_L2")
            wires_VSI_L2Rd.x_min = block_instr_L1.x_max + 4
            wires_VSI_L2Rd.x_max = wires_VSI_L2Rd.x_min + 14
            wires_VSI_L2Rd.y_min = block_instr_L1.y_min
            wires_VSI_L2Rd.y_max = block_vector_L1.y_max
            wires_VSI_L2Rd.vl1_l2_rd = mem_chart_data["VL1_L2 Rd"]
            wires_VSI_L2Rd.vl1_l2_wr = mem_chart_data["VL1_L2 Wr"]
            wires_VSI_L2Rd.vl1_l2_atomic = mem_chart_data["VL1_L2 Atomic"]
            wires_VSI_L2Rd.constL1_L2_rd = mem_chart_data["constL1_L2_rd"]
            wires_VSI_L2Rd.constL1_L2_wr = mem_chart_data["constL1_L2_wr"]
            wires_VSI_L2Rd.constL1_L2_atomic = mem_chart_data["constL1_L2_atomic"]
            wires_VSI_L2Rd.instrL1_L2_req = mem_chart_data["instrL1_L2_req"]

            wires_VSI_L2Rd.draw(canvas)

            # ----------------------------------------
            # L2 Cache block
            block_L2 = L2Cache(label="L2 Cache")

            block_L2.x_min = wires_VSI_L2Rd.x_max + 1
            block_L2.x_max = block_L2.x_min + 24
            block_L2.y_min = block_instr_L1.y_min
            block_L2.y_max = block_lds.y_max

            block_L2.rd = mem_chart_data["L2 Rd"]
            block_L2.wr = mem_chart_data["L2 Wr"]
            block_L2.atomic = mem_chart_data["L2 Atomic"]
            block_L2.hit = mem_chart_data["L2 Hit"]
            block_L2.rd_cycles = mem_chart_data["L2 Rd Lat"]
            block_L2.wr_cycles = mem_chart_data["L2 Wr Lat"]

            block_L2.draw(canvas)

            # ----------------------------------------
            # Wires between L2 block and Fabric
            wires_L2_Fabric = Wire_L2_Fabric(label="Wire_L2_Fabric",
                                    x_min=block_L2.x_max + 1,
                                    x_max=block_L2.x_max + 16,
                                    y_min=block_L2.y_max - 18,
                                    y_max=block_L2.y_max - 10)

            wires_L2_Fabric.rd = mem_chart_data["L2_Fabric_rd"]
            wires_L2_Fabric.wr = mem_chart_data["L2_Fabric_wr"]
            wires_L2_Fabric.atomic = mem_chart_data["L2_Fabric_atomic"]

            wires_L2_Fabric.draw(canvas)

            # ----------------------------------------
            # xGMI/PCIe block with wires to fabric
            block_xgmi_pcie = xGMI_PCIe(label="xGMI/PCIe",
                                x_min=wires_L2_Fabric.x_max + 10,
                                x_max=wires_L2_Fabric.x_max + 20,
                                y_min=block_L2.y_max - 4,
                                y_max=block_L2.y_max)
            block_xgmi_pcie.draw(canvas)

            # ----------------------------------------
            # Data Fabric Block
            block_fabric = Fabric(label="Fabric",
                            x_min=wires_L2_Fabric.x_max + 3,
                            x_max=wires_L2_Fabric.x_max + 27,
                            y_max=block_xgmi_pcie.y_min - 5,
                            y_min=block_xgmi_pcie.y_min - 5 - 11)
            block_fabric.draw(canvas)

            # ----------------------------------------
            # GMI block with wires to fabric
            block_gmi = GMI(label="GMI",
                            x_min=block_xgmi_pcie.x_min,
                            x_max=block_xgmi_pcie.x_max,
                            y_min=block_fabric.y_min - 9,
                            y_max=block_fabric.y_min - 5)
            block_gmi.draw(canvas)

            # ----------------------------------------
            # Wires between fabric and HBM
            # Wire_Fabric_HBM
            wires_Fabric_HBM = Wire_Fabric_HBM(label="Wire_Fabric_HBM",
                                    x_min=block_fabric.x_max + 1,
                                    x_max=block_fabric.x_max + 15,
                                    y_min=block_fabric.y_max - 2,
                                    y_max=block_fabric.y_max - 4)

            # fixme
            #wires_Fabric_HBM.rd = mem_chart_data["L2_Fabric_rd"]
            #wires_Fabric_HBM.wr = mem_chart_data["L2_Fabric_wr"]

            wires_Fabric_HBM.draw(canvas)

            # ----------------------------------------
            # HBM block
            block_hbm = HBM(label="HBM",
                            x_min=wires_Fabric_HBM.x_max,
                            x_max=wires_Fabric_HBM.x_max + 10,
                            y_min=block_fabric.y_max - 7,
                            y_max=block_fabric.y_max - 3)
            block_hbm.draw(canvas)

    canvas = Canvas(width=234, height=42, xmax=234, ymax=42)

    mc = MemChart(0, 0, 233, 41)
    mc.draw(canvas, mem_chart_data)

    # return the plot string stream
    return canvas.plot()


if __name__ == '__main__':
    df = ''
    arch = ''
    print(plot_mem_chart(df, arch))
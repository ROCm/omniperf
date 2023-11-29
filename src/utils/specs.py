"""Get host/gpu specs."""

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

import os
import re
import sys
import socket
import subprocess

from dataclasses import dataclass
from pathlib import Path as path
from textwrap import dedent


@dataclass
class MachineSpecs:
    hostname: str
    CPU: str
    kernel_version: str
    ram: str
    distro: str
    rocm_version: str
    GPU: str
    arch: str
    L1: str
    L2: str
    CU: str
    SIMD: str
    SE: str
    wave_size: str
    workgroup_max_size: str
    max_sclk: str
    cur_sclk: str
    cur_mclk: str
    max_waves_per_cu: str
    L2Banks: str
    LDSBanks: str
    numSQC: str
    hbmBW: str

    def __str__(self):
        return dedent(
            f"""\
        Host info:
            hostname:           {self.hostname}
            CPU:                {self.CPU}
            ram:                {self.ram}
            distro:             {self.distro}
            kernel_version:     {self.kernel_version}
            rocm_version:       {self.rocm_version}
        Device info:
            GPU:                {self.GPU}
            arch:               {self.arch}
            L1:                 {self.L1} KB
            L2:                 {self.L2} KB
            max_sclk:           {self.max_sclk} MHz
            cur_sclk:           {self.cur_sclk} MHz
            cur_mclk:           {self.cur_mclk} MHz
            CU:                 {self.CU}
            SIMD:               {self.SIMD}
            SE:                 {self.SE}
            wave_size:          {self.wave_size}
            workgroup_max_size: {self.workgroup_max_size}
            max_waves_per_cu:   {self.max_waves_per_cu}
            L2Banks:            {self.L2Banks}
            LDSBanks:           {self.LDSBanks}
            numSQC:             {self.numSQC}
            hbmBW:              {self.hbmBW} MB/s
        """
        )


def gpuinfo():
    # Local var only for rocminfo searching
    gpu_list = {"gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942"}

    # Fixme: find better way to differentiate cards, GPU vs APU, etc.

    rocminfo = run(["rocminfo"]).split("\n")

    for idx1, linetext in enumerate(rocminfo):
        gpu_id = search(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", linetext)
        if gpu_id in gpu_list:
            break
        if str(gpu_id) in gpu_list:
            gpu_id = str(gpu_id)
            break

    if not gpu_id in gpu_list:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    L1, L2 = "", ""
    for idx2, linetext in enumerate(rocminfo[idx1 + 1 :]):
        key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            L1 = key
            continue

        key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            L2 = key
            continue

        key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
        if key != None:
            max_sclk = key
            continue

        key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_CU = key
            continue

        key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_SIMD = key
            continue

        key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            num_SE = key
            continue

        key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            wave_size = key
            continue

        key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            grp_size = key
            continue

        key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
        if key != None:
            max_waves_per_cu = key
            break

    gpu_name = ""
    L2Banks = ""
    LDSBanks = "32"
    numSQC = ""

    if gpu_id == "gfx906":
        gpu_name = "mi50"
        L2Banks = "16"
        numSQC = str(int(num_CU) // 4)
    elif gpu_id == "gfx908":
        gpu_name = "mi100"
        L2Banks = "32"
        numSQC = "48"
    elif gpu_id == "gfx90a":
        L2Banks = "32"
        gpu_name = "mi200"
        numSQC = "56"
    elif gpu_id == "gfx940":
        gpu_name = "mi300A_A0"
        L2Banks = "16"
        numSQC = "56"
    elif gpu_id == "gfx941":
        gpu_name = "mi300X_A0"
        L2Banks = "16"
        numSQC = "56"
    elif (gpu_id == "gfx942") and ("MI300A" in rocminfo):
        gpu_name = "mi300A_A1"
        L2Banks = "16"
        numSQC = "56"
    elif (gpu_id == "gfx942") and ("MI300A" not in rocminfo):
        gpu_name = "mi300X_A1"
        L2Banks = "16"
        numSQC = "56"

    return (
        gpu_name,
        gpu_id,
        L1,
        L2,
        max_sclk,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
    )


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd[0] == "rocm-smi" and p.returncode == 8:
        print("ERROR: No GPU detected. Unable to load rocm-smi")
        sys.exit(1)
    return p.stdout.decode("utf-8")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def get_machine_specs(devicenum):
    cpuinfo = path("/proc/cpuinfo").read_text()
    meminfo = path("/proc/meminfo").read_text()
    version = path("/proc/version").read_text()
    os_release = path("/etc/os-release").read_text()

    version_loc = [
        "version",
        "version-dev",
        "version-hip-libraries",
        "version-hiprt",
        "version-hiprt-devel",
        "version-hip-sdk",
        "version-libs",
        "version-utils",
    ]

    rocmFound = False
    for itr in version_loc:
        _path = os.path.join(os.getenv("ROCM_PATH", "/opt/rocm"), ".info", itr)
        if os.path.exists(_path):
            rocm_ver = path(_path).read_text()
            rocmFound = True
            break

    if not rocmFound:
        # check if ROCM_VER is supplied externally
        ROCM_VER_USER = os.getenv("ROCM_VER")
        if ROCM_VER_USER is not None:
            print(
                "Overriding missing ROCm version detection with ROCM_VER = %s"
                % ROCM_VER_USER
            )
            rocm_ver = ROCM_VER_USER
        else:
            _rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            print("Error: Unable to detect a complete local ROCm installation.")
            print(
                "\nThe expected %s/.info/ versioning directory is missing. Please"
                % _rocm_path
            )
            print("ensure you have valid ROCm installation.")
            sys.exit(1)

    (
        gpu_name,
        gpu_id,
        L1,
        L2,
        max_sclk,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
    ) = gpuinfo()

    rocm_smi = run(["rocm-smi"])

    # # Clean rocm_smi
    # rocm_smi_raw = run(["rocm-smi"]).splitlines()
    # rocm_smi = []
    # for row in rocm_smi_raw:
    #     splt = row.split()
    #     if row and not row.startswith("=") and splt[0].find("[") == -1:
    #         # note this will also create an entry for header
    #         rocm_smi.append(splt)

    # rocm_smi[0].remove("(DieEdge)") # remove superfluous headers

    # smi_dict = {}
    # for header_idx in range(0, len(rocm_smi[0])):
    #     header = rocm_smi[0][header_idx]
    #     smi_dict[header] = []
    #     # Loop over each gpu enty
    #     for row in range(1, len(rocm_smi)):
    #         # verify it has all expected fields
    #         if(len(rocm_smi[0]) != len(rocm_smi[row])):
    #             sys.exit(1)
    #         # push into dict if starts w a num
    #         if search(r"([0-9]+)", rocm_smi[row][0]):
    #             smi_dict[header].append(rocm_smi[row][header_idx])

    device = rf"^\s*{devicenum}(.*)"

    hostname = socket.gethostname()
    CPU = search(r"^model name\s*: (.*?)$", cpuinfo)
    kernel_version = search(r"version (\S*)", version)
    ram = search(r"MemTotal:\s*(\S*)", meminfo)
    distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    if distro is None:
        distro = ""

    rocm_version = rocm_ver.strip()

    freq = search(device, rocm_smi).split()
    cur_sclk = search(r"([0-9]+)", freq[2])
    if cur_sclk is None:
        cur_sclk = ""

    cur_mclk = search(r"([0-9]+)", freq[3])
    if cur_mclk is None:
        cur_mclk = 0

    # FIXME with spec
    hbmBW = str(int(cur_mclk) / 1000 * 4096 / 8 * 2)

    return MachineSpecs(
        hostname,
        CPU,
        kernel_version,
        ram,
        distro,
        rocm_version,
        gpu_name,
        gpu_id,
        L1,
        L2,
        max_sclk,
        num_CU,
        num_SIMD,
        num_SE,
        wave_size,
        grp_size,
        cur_sclk,
        cur_mclk,
        max_waves_per_cu,
        L2Banks,
        LDSBanks,
        numSQC,
        hbmBW,
    )


if __name__ == "__main__":
    print(get_machine_specs(0))

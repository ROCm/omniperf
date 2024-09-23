"""Get host/gpu specs."""

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
import socket
import subprocess
import importlib
import config
import pandas as pd

from datetime import datetime
from math import ceil
from dataclasses import dataclass, field, fields
from pathlib import Path as path
from utils.utils import (
    total_xcds,
    get_version,
    console_error,
    console_warning,
    console_log,
)
from utils.tty import get_table_string

VERSION_LOC = [
    "version",
    "version-dev",
    "version-hip-libraries",
    "version-hiprt",
    "version-hiprt-devel",
    "version-hip-sdk",
    "version-libs",
    "version-utils",
]


def detect_arch(_rocminfo):
    from omniperf_base import SUPPORTED_ARCHS

    for idx1, linetext in enumerate(_rocminfo):
        gpu_arch = search(r"^\s*Name\s*:\s+ ([a-zA-Z0-9]+)\s*$", linetext)
        if gpu_arch in SUPPORTED_ARCHS.keys():
            break
        if str(gpu_arch) in SUPPORTED_ARCHS.keys():
            gpu_arch = str(gpu_arch)
            break
    if not gpu_arch in SUPPORTED_ARCHS.keys():
        console_error("Cannot find a supported arch in rocminfo")
    else:
        return (gpu_arch, idx1)


# Custom decorator to mimic the behavior of kw_only found in Python 3.10
def kw_only(cls):
    def __init__(self, *args, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    cls.__init__ = __init__
    return cls


def generate_machine_specs(args, sysinfo: dict = None):
    if not sysinfo is None:
        try:
            sysinfo_ver = str(sysinfo["version"])
        except KeyError:
            console_error(
                "Detected mismatch in sysinfo versioning. You need to reprofile to update data."
            )
        version = get_version(config.omniperf_home)["version"]
        if sysinfo_ver != version[: version.find(".")]:
            console_error(
                "Detected mismatch in sysinfo versioning. You need to reprofile to update data."
            )
        return MachineSpecs(**sysinfo)
    # read timestamp info
    now = datetime.now()
    local_now = now.astimezone()
    local_tz = local_now.tzinfo
    local_tzname = local_tz.tzname(local_now)
    timestamp = now.strftime("%c") + " (" + local_tzname + ")"
    hostname = socket.gethostname()

    # set specs version
    vData = get_version(config.omniperf_home)
    version = vData["version"]
    # NB: Just taking major as specs version. May want to make this more specific in the future
    specs_version = version[
        : version.find(".")
    ]  # version will always follow 'major.minor.patch' format

    ##########################################
    ## A. Machine Specs
    ##########################################
    cpuinfo = path("/proc/cpuinfo").read_text()
    meminfo = path("/proc/meminfo").read_text()
    version = path("/proc/version").read_text()
    os_release = path("/etc/os-release").read_text()
    cpu_model = search(r"^model name\s*: (.*?)$", cpuinfo)
    sbios = (
        path("/sys/class/dmi/id/bios_vendor").read_text().strip()
        + path("/sys/class/dmi/id/bios_version").read_text().strip()
    )
    linux_kernel_version = search(r"version (\S*)", version)
    amd_gpu_kernel_version = ""  # TODO: Extract amdgpu kernel version
    cpu_memory = search(r"MemTotal:\s*(\S*)", meminfo)
    gpu_memory = ""  # TODO: Extract gpu memory
    linux_distro = search(r'PRETTY_NAME="(.*?)"', os_release)
    if linux_distro is None:
        linux_distro = ""
    rocm_version = get_rocm_ver().strip()
    # FIXME: use device
    vbios = search(r"VBIOS version: (.*?)$", run(["rocm-smi", "-v"], exit_on_error=True))
    compute_partition = search(
        r"Compute Partition:\s*(\w+)", run(["rocm-smi", "--showcomputepartition"])
    )
    if compute_partition is None:
        compute_partition = "NA"
    memory_partition = search(
        r"Memory Partition:\s*(\w+)", run(["rocm-smi", "--showmemorypartition"])
    )
    if memory_partition is None:
        memory_partition = "NA"

    ##########################################
    ## B. SoC Specs
    ##########################################
    # read rocminfo
    rocminfo_full = run(["rocminfo"])
    _rocminfo = rocminfo_full.split("\n")
    gpu_arch, idx = detect_arch(_rocminfo)
    _rocminfo = _rocminfo[idx + 1 :]  # update rocminfo for target section
    specs = MachineSpecs(
        version=specs_version,
        timestamp=timestamp,
        _rocminfo=_rocminfo,
        hostname=hostname,
        cpu_model=cpu_model,
        sbios=sbios,
        linux_kernel_version=linux_kernel_version,
        amd_gpu_kernel_version=amd_gpu_kernel_version,
        cpu_memory=cpu_memory,
        gpu_memory=gpu_memory,
        linux_distro=linux_distro,
        rocm_version=rocm_version,
        vbios=vbios,
        compute_partition=compute_partition,
        memory_partition=memory_partition,
        gpu_arch=gpu_arch,
    )
    # Load above SoC specs via module import
    try:
        soc_module = importlib.import_module("omniperf_soc.soc_" + specs.gpu_arch)
    except ModuleNotFoundError as e:
        console_error(
            "Arch %s marked as supported, but couldn't find class implementation %s."
            % (specs.gpu_arch, e)
        )
    soc_class = getattr(soc_module, specs.gpu_arch + "_soc")
    soc_obj = soc_class(args, specs)
    # Update arch specific specs
    specs.total_l2_chan: str = total_l2_banks(
        specs.gpu_model, int(specs._l2_banks), specs.compute_partition
    )
    specs.hbm_bw: str = str(int(specs.max_mclk) / 1000 * 32 * specs.get_hbm_channels())
    return specs


@kw_only
@dataclass
class MachineSpecs:
    ##########################################
    ## A. Workload / Spec info
    ##########################################

    # these three fields are special in that they're not included
    # when you use (e.g.,) --specs to view the machinespecs, but they
    # _are_ included in profiling/analysis, so we mark them as 'optional'
    # in the metadata to avoid erroring out on missing fields on
    # serialization
    workload_name: str = field(
        default=None,
        metadata={
            "doc": "The name of the workload data was collected for.",
            "name": "Workload Name",
            "optional": True,
        },
    )
    command: str = field(
        default=None,
        metadata={
            "doc": "The command the workload was executed with.",
            "name": "Command",
            "optional": True,
        },
    )
    ip_blocks: str = field(
        default=None,
        metadata={
            "doc": "The hardware blocks profiling information was collected for.",
            "name": "IP Blocks",
            "optional": True,
        },
    )
    timestamp: str = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
        },
    )
    version: str = field(
        default=None,
        metadata={
            "doc": "The version of the machine specification file format.",
            "name": "MachineSpecs Version",
            "intable": False,
        },
    )
    timestamp: str = field(
        default=None,
        metadata={
            "doc": "The time (in local system time) when data was collected",
            "name": "Timestamp",
        },
    )
    _rocminfo: list = field(default=None)
    ##########################################
    ## A. Machine Specs
    ##########################################
    hostname: str = field(
        default=None, metadata={"doc": "The hostname of the machine.", "name": "Hostname"}
    )
    cpu_model: str = field(
        default=None,
        metadata={"doc": "The model name of the CPU used.", "name": "CPU Model"},
    )
    sbios: str = field(
        default=None,
        metadata={
            "doc": "The system management bios version and vendor.",
            "name": "SBIOS",
        },
    )
    linux_distro: str = field(
        default=None,
        metadata={
            "doc": "The Linux distribution installed on the machine.",
            "name": "Linux Distribution",
        },
    )
    linux_kernel_version: str = field(
        default=None,
        metadata={
            "doc": "The Linux kernel version running on the machine.",
            "name": "Linux Kernel Version",
        },
    )
    amd_gpu_kernel_version: str = field(
        default=None,
        metadata={
            "doc": "[RESERVED] The version of the AMDGPU driver installed on the machine. Unimplemented.",
            "name": "AMD GPU Kernel Version",
        },
    )
    cpu_memory: str = field(
        default=None,
        metadata={
            "doc": "The total amount of memory available to the CPU.",
            "unit": "KB",
            "name": "CPU Memory",
        },
    )
    gpu_memory: str = field(
        default=None,
        metadata={
            "doc": "[RESERVED] The total amount of memory available to accelerators/GPUs in the system. Unimplemented.",
            "unit": "KB",
            "name": "GPU Memory",
        },
    )
    rocm_version: str = field(
        default=None,
        metadata={
            "doc": "The ROCm version used during data-collection.",
            "name": "ROCm Version",
        },
    )
    vbios: str = field(
        default=None,
        metadata={
            "doc": "The version of the accelerators/GPUs video bios in the system.",
            "name": "VBIOS",
        },
    )
    compute_partition: str = field(
        default=None,
        metadata={
            "doc": "The compute partitioning mode active on the accelerators/GPUs in the system (MI300 only).",
            "name": "Compute Partition",
        },
    )
    memory_partition: str = field(
        default=None,
        metadata={
            "doc": "The memory partitioning mode active on the accelerators/GPUs in the system (MI300 only).",
            "name": "Memory Partition",
        },
    )

    ##########################################
    ## B. SoC Specs
    ##########################################
    gpu_model: str = field(
        default=None,
        metadata={
            "doc": "The product name of the accelerators/GPUs in the system.",
            "name": "GPU Model",
        },
    )
    gpu_arch: str = field(
        default=None,
        metadata={
            "doc": "The architecture name of the accelerators/GPUs in the system,\n"
            "as used by (e.g.,) the AMDGPU backed of LLVM.",
            "name": "GPU Arch",
        },
    )
    gpu_l1: str = field(
        default=None,
        metadata={
            "doc": "The size of the vL1D cache (per compute-unit) on the accelerators/GPUs.",
            "name": "GPU L1",
            "unit": "KiB",
        },
    )
    gpu_l2: str = field(
        default=None,
        metadata={
            "doc": "The size of the vL1D cache (per compute-unit) on the accelerators/GPUs.",
            "name": "GPU L2",
            "unit": "KiB",
        },
    )
    cu_per_gpu: str = field(
        default=None,
        metadata={
            "doc": "The total number of compute units per accelerator/GPU in the system. On systems with configurable\n"
            "partitioning, (e.g., MI300) this is the total number of compute units in a partition.",
            "name": "CU per GPU",
        },
    )
    simd_per_cu: str = field(
        default=None,
        metadata={
            "doc": "The number of SIMD processors in a compute unit for the accelerators/GPUs in the system.",
            "name": "SIMD per CU",
        },
    )
    se_per_gpu: str = field(
        default=None,
        metadata={
            "doc": "The number of shader engines on the accelerators/GPUs in the system. On systems with configurable\n"
            "partitioning, (e.g., MI300) this is the total number of shader engines in a partition.",
            "name": "SE per GPU",
        },
    )
    wave_size: str = field(
        default=None,
        metadata={
            "doc": "The number work-items in a wavefront on the accelerators/GPUs in the system.",
            "name": "Wave Size",
        },
    )
    workgroup_max_size: str = field(
        default=None,
        metadata={
            "doc": "The maximum number of work-items in a workgroup on the accelerators/GPUs in the system.",
            "name": "Workgroup Max Size",
        },
    )
    chip_id: str = field(
        default=None,
        metadata={
            "doc": "<>",
            "name": "Chip ID",
        },
    )
    max_waves_per_cu: str = field(
        default=None,
        metadata={
            "doc": "The maximum number of wavefronts that can be resident on a compute unit on the\n"
            "accelerators/GPUs in the system",
            "name": "Max Waves per CU",
        },
    )
    max_sclk: str = field(
        default=None,
        metadata={
            "doc": "The maximum engine (compute-unit) clock rate of the accelerators/GPUs in the system.",
            "name": "Max SCLK",
            "unit": "MHz",
        },
    )
    max_mclk: str = field(
        default=None,
        metadata={
            "doc": "The maximum memory clock rate of the accelerators/GPUs in the system.",
            "name": "Max MCLK",
            "unit": "MHz",
        },
    )
    cur_sclk: str = field(
        default=None,
        metadata={
            "doc": "[RESERVED] The current engine (compute unit) clock rate of the accelerators/GPUs in the system. Unused.",
            "name": "Cur SCLK",
            "unit": "MHz",
        },
    )
    cur_mclk: str = field(
        default=None,
        metadata={
            "doc": "[RESERVED] The current memory clock rate of the accelerators/GPUs in the system. Unused.",
            "name": "Cur MCLK",
            "unit": "MHz",
        },
    )
    _l2_banks: str = None  # NB: This only used in flatten_tcc_info_across_hbm_stacks()
    total_l2_chan: str = field(
        default=None,
        metadata={
            "doc": "The maximum number of L2 cache channels on the accelerators/GPUs in the system. On systems with\n"
            "configurable partitioning, (e.g., MI300) this is the total number of L2 cache channels in a partition.",
            "name": "Total L2 Channels",
        },
    )
    lds_banks_per_cu: str = field(
        default=None,
        metadata={
            "doc": "The number of banks in the LDS for a compute unit on the accelerators/GPUs in the system.",
            "name": "LDS Banks per CU",
        },
    )
    sqc_per_gpu: str = field(
        default=None,
        metadata={
            "doc": "The number of L1I/sL1D caches on the accelerators/GPUs in the system. On systems with\n"
            "configurable partitioning, (e.g., MI300) this is the total number of L1I/sL1D caches in a partition.",
            "name": "SQC per GPU",
        },
    )
    pipes_per_gpu: str = field(
        default=None,
        metadata={
            "doc": "The number of scheduler-pipes on the accelerators/GPUs in the system.",
            "name": "Pipes per GPU",
        },
    )
    hbm_bw: str = field(
        default=None,
        metadata={
            "doc": "The peak theoretical HBM bandwidth for the accelerators/GPUs in the system. On systems with\n"
            "configurable partitioning, (e.g., MI300) this is the peak theoretical HBM bandwidth for a partition.",
            "name": "HBM BW",
            "unit": "GB/s",
        },
    )
    num_xcd: str = field(
        default=None,
        metadata={
            "doc": "The total number of accelerator complex dies in a compute partition on the accelerators/GPUs in the\n"
            "system.  For accelerators without partitioning (i.e., pre-MI300), this is considered to be one.",
            "name": "Num XCDs",
            "unit": "XCDs",
        },
    )

    def get_hbm_channels(self):
        # check MI300 has a valid compute partition
        mi300a_archs = ["mi300a_a0", "mi300a_a1"]
        mi300x_archs = ["mi300x_a0", "mi300x_a1"]
        mi308x_archs = ["mi308x"]
        if self.gpu_model.lower() in mi300a_archs + mi300x_archs + mi308x_archs:
            hbmchannels = 128
            if self.memory_partition.lower() == "nps2":
                hbmchannels /= 2
            elif self.memory_partition.lower() == "nps4":
                hbmchannels /= 4
            elif self.memory_partition.lower() == "nps8":
                hbmchannels /= 8
            return int(hbmchannels)
        else:
            hbmchannels = int(self.total_l2_chan)
        return hbmchannels

    def get_class_members(self):
        all_populated = True
        data = {}
        # dataclass uses an OrderedDict for member variables, ensuring order consistency
        for field in fields(self):
            name = field.name
            if not name.startswith("_"):
                value = getattr(self, name)
                if value is None:
                    # check if we've marked it optional
                    if (
                        field.metadata
                        and "optional" in field.metadata
                        and field.metadata["optional"]
                    ):
                        pass
                    else:
                        console_warning(
                            f"Incomplete class definition for {self.gpu_arch}. "
                            f"Expecting populated {name} but detected None."
                        )
                        all_populated = False
                data[name] = value

        if not all_populated:
            console_error("Missing specs fields for %s" % self.gpu_arch)
        return pd.DataFrame(data, index=[0])

    def __repr__(self):
        topstr = "Machine Specifications: describing the state of the machine that Omniperf data was collected on.\n"
        data = []
        for field in fields(self):
            name = field.name
            if not name.startswith("_"):
                _data = {}
                value = getattr(self, name)
                if field.metadata:
                    # check out of table before any re-naming for pretty-printing
                    if "intable" in field.metadata and not field.metadata["intable"]:
                        if name == "version":
                            topstr += f"Output version: {value}\n"
                        else:
                            console_error(f"Unknown out of table printing field: {name}")
                        continue
                    if "name" in field.metadata:
                        name = field.metadata["name"]
                    if "unit" in field.metadata:
                        _data["Unit"] = field.metadata["unit"]
                    if "doc" in field.metadata:
                        _data["Description"] = field.metadata["doc"]
                _data["Spec"] = name
                _data["Value"] = value
                data.append(_data)
        df = pd.DataFrame(data)
        columns = ["Spec", "Value"]
        if "Description" in df.columns:
            columns += ["Description"]
        if "Unit" in df.columns:
            columns += ["Unit"]
        df = df[columns]
        df = df.fillna("")
        return topstr + get_table_string(df, transpose=False, decimal=2)


def get_rocm_ver():
    rocm_found = False
    for itr in VERSION_LOC:
        _path = os.path.join(os.getenv("ROCM_PATH", "/opt/rocm"), ".info", itr)
        if os.path.exists(_path):
            rocm_ver = path(_path).read_text()
            rocm_found = True
        break
    if not rocm_found:
        # check if ROCM_VER is supplied externally
        ROCM_VER_USER = os.getenv("ROCM_VER")
        if ROCM_VER_USER is not None:
            console_log(
                "profiling",
                "Overriding missing ROCm version detection with ROCM_VER = %s"
                % ROCM_VER_USER,
            )
            rocm_ver = ROCM_VER_USER
        else:
            _rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
            console_warning("Unable to detect a complete local ROCm installation.")
            console_warning(
                "The expected %s/.info/ versioning directory is missing." % _rocm_path
            )
            console_error("Ensure you have valid ROCm installation.")
    return rocm_ver


def run(cmd, exit_on_error=False):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        console_error(
            f"Unable to parse specs. Can't find ROCm asset: {e.filename}\nTry passing a path to an existing workload results in 'analyze' mode."
        )

    if exit_on_error:
        if cmd[0] == "rocm-smi":
            if p.returncode != 2 and p.returncode != 0:
                console_error("No GPU detected. Unable to load rocm-smi")
        elif p.returncode != 0:
            console_error("Command [%s] failed with non-zero exit code" % cmd)
    return p.stdout.decode("utf-8")


def search(pattern, string):
    m = re.search(pattern, string, re.MULTILINE)
    if m is not None:
        return m.group(1)
    return None


def total_sqc(archname, numCUs, numSEs):
    cu_per_se = float(numCUs) / float(numSEs)
    sq_per_se = cu_per_se / 2
    if archname.lower() in ["mi50", "mi100"]:
        sq_per_se = cu_per_se / 3
    sq_per_se = ceil(sq_per_se)
    return int(sq_per_se) * int(numSEs)


def total_l2_banks(archname, L2Banks, compute_partition):
    # Fixme: support all supported partitioning mode
    # Fixme: "name" is a bad name!
    totalL2Banks = L2Banks
    xcds = total_xcds(archname, compute_partition)
    return L2Banks * xcds


if __name__ == "__main__":
    print(generate_machine_specs())

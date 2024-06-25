.. _pipeline-metrics:

Pipeline metrics
================

In this section, we describe the metrics available in Omniperf to analyze the
pipelines discussed in the :ref:`pipeline-desc`.

.. _wavefront:

Wavefront
---------

.. _wavefront-launch-stats:

Wavefront launch stats
^^^^^^^^^^^^^^^^^^^^^^

The wavefront launch stats panel gives general information about the
kernel launch:

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric
     - Description
     - Unit
   * - Grid Size
     - The total number of work-items (a.k.a "threads") launched as a part of the kernel dispatch.  In HIP, this is equivalent to the total grid size multiplied by the total workgroup (a.k.a "block") size.
     - [Work-items](Workitem)
   * - Workgroup Size
     - The total number of work-items (a.k.a "threads") in each workgroup (a.k.a "block") launched as part of the kernel dispatch.  In HIP, this is equivalent to the total block size.
     - [Work-items](Workitem)
   * - Total Wavefronts
     - The total number of wavefronts launched as part of the kernel dispatch.  On AMD Instinct(tm) CDNA accelerators and GCN GPUs, the wavefront size is always 64 work-items.  Thus, the total number of wavefronts should be equivalent to the ceiling of Grid Size divided by 64.
     - [Wavefronts](Wavefront)
   * - Saved Wavefronts
     - The total number of wavefronts saved at a context-save, see [cwsr_enable](https://docs.kernel.org/gpu/amdgpu/module-parameters.html?highlight=cwsr).
     - [Wavefronts](Wavefront)
   * - Restored Wavefronts
     - The total number of wavefronts restored from a context-save, see [cwsr_enable](https://docs.kernel.org/gpu/amdgpu/module-parameters.html?highlight=cwsr).
     - [Wavefronts](Wavefront)
   * - VGPRs
     - The number of architected vector general-purpose registers allocated for the kernel, see [VALU](valu).  Note: this may not exactly match the number of VGPRs requested by the compiler due to allocation granularity.
     - [VGPRs](valu)
   * - AGPRs
     - The number of accumulation vector general-purpose registers allocated for the kernel, see [AGPRs](agprs).  Note: this may not exactly match the number of AGPRs requested by the compiler due to allocation granularity.
     - [AGPRs](agprs)
   * - SGPRs
     - The number of scalar general-purpose registers allocated for the kernel, see [SALU](salu).  Note: this may not exactly match the number of SGPRs requested by the compiler due to allocation granularity.
     - [SGPRs](salu)
   * - LDS Allocation
     - The number of bytes of [LDS](lds) memory (a.k.a., "Shared" memory) allocated for this kernel.  Note: This may also be larger than what was requested at compile-time due to both allocation granularity and dynamic per-dispatch LDS allocations.
     - Bytes per [workgroup](workgroup)
   * - Scratch Allocation
     - The number of bytes of [scratch-memory](Mspace) requested _per_ work-item for this kernel.  Scratch memory is used for stack memory on the accelerator, as well as for register spills/restores.
     - Bytes per [work-item](workitem)

.. _wavefront-runtime-stats:

Wavefront Runtime Stats
^^^^^^^^^^^^^^^^^^^^^^^

The wavefront runtime statistics gives a high-level overview of the
execution of wavefronts in a kernel:

.. list-table::
   :header-rows: 1
   :widths: 18 65 17

   * - Metric
     - Description
     - Unit
   * - [Kernel Time](KernelTime)
     - The total duration of the executed kernel.  Note: this should not be directly compared to the wavefront cycles / timings below.
     - Nanoseconds
   * - [Kernel Cycles](KernelCycles)
     - The total duration of the executed kernel in cycles.  Note: this should not be directly compared to the wavefront cycles / timings below.
     - Cycles
   * - Instructions per wavefront
     - The average number of instructions (of all types) executed per wavefront.  This is averaged over all wavefronts in a kernel dispatch.
     - Instructions / wavefront
   * - Wave Cycles
     - The number of cycles a wavefront in the kernel dispatch spent resident on a compute unit per [normalization-unit](normunit).  This is averaged over all wavefronts in a kernel dispatch.  Note: this should not be directly compared to the kernel cycles above.
     - Cycles per [normalization-unit](normunit)
   * - Dependency Wait Cycles
     - The number of cycles a wavefront in the kernel dispatch stalled waiting on memory of any kind (e.g., instruction fetch, vector or scalar memory, etc.) per [normalization-unit](normunit).  This counter is incremented at every cycle by _all_ wavefronts on a CU stalled at a memory operation.  As such, it is most useful to get a sense of how waves were spending their time, rather than identification of a precise limiter because another wave could be actively executing while a wave is stalled.  The sum of this metric, Issue Wait Cycles and Active Cycles should be equal to the total Wave Cycles metric.
     - Cycles per [normalization-unit](normunit)
   * - Issue Wait Cycles
     - The number of cycles a wavefront in the kernel dispatch was unable to issue an instruction for any reason (e.g., execution pipe back-pressure, arbitration loss, etc.) per [normalization-unit](normunit).  This counter is incremented at every cycle by _all_ wavefronts on a CU unable to issue an instruction.  As such, it is most useful to get a sense of how waves were spending their time, rather than identification of a precise limiter because another wave could be actively executing while a wave is issue stalled.  The sum of this metric, Dependency Wait Cycles and Active Cycles should be equal to the total Wave Cycles metric.
     - Cycles per [normalization-unit](normunit)
   * - Active Cycles
     - The average number of cycles a wavefront in the kernel dispatch was actively executing instructions per [normalization-unit](normunit).  This measurement is made on a per-wavefront basis, and may include (e.g.,) cycles that another wavefront spent actively executing (e.g., on another execution unit) or was stalled.  As such, it is most useful to get a sense of how waves were spending their time, rather than identification of a precise limiter.  The sum of this metric, Issue Wait Cycles and Active Wait Cycles should be equal to the total Wave Cycles metric.
     - Cycles per [normalization-unit](normunit)
   * - Wavefront Occupancy
     - The time-averaged number of wavefronts resident on the accelerator over the lifetime of the kernel. Note: this metric may be inaccurate for short-running kernels (<< 1ms).
     - Wavefronts

.. code:: {seealso}

   As mentioned above, the measurement of kernel cycles and time typically cannot directly be compared to e.g., Wave Cycles.
   This is due to two factors: first, the kernel cycles/timings are measured using a counter that is impacted by scheduling overhead, this is particularly noticeable for "short-running" kernels (typically << 1ms) where scheduling overhead forms a significant portion of the overall kernel runtime.
   Secondly, the Wave Cycles metric is incremented per-wavefront scheduled to a SIMD every cycle whereas the kernel cycles counter is incremented only once per-cycle when _any_ wavefront is scheduled.

.. _instruction-mix:

Instruction Mix
---------------

The instruction mix panel shows a breakdown of the various types of
instructions executed by the user’s kernel, and which pipelines on the
`CU <CU>`__ they were executed on. In addition, Omniperf reports further
information about the breakdown of operation types for the
`VALU <valu>`__, vector-memory, and `MFMA <mfma>`__ instructions.

.. code:: {note}

   All metrics in this section count _instructions issued_, and _not_ the total number of operations executed.
   The values reported by these metrics will not change regardless of the execution mask of the wavefront.
   We note that even if the execution mask is identically zero (i.e., _no lanes are active_) the instruction will still be counted, as CDNA accelerators still consider these instructions 'issued' see, e.g., [EXECute Mask, Section 3.3 of the CDNA2 ISA Guide](https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf) for more details.

Overall Instruction Mix
^^^^^^^^^^^^^^^^^^^^^^^

This panel shows the total number of each type of instruction issued to
the :ref:`various compute pipelines <pipeline-desc>` on the
:ref:`CU <compute-unit>`. These are:

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric
     - Description
     - Unit
   * - [VALU](valu) Instructions
     - The total number of vector arithmetic logic unit (VALU) operations issued. These are the workhorses of the compute-unit, and are used to execute wide range of instruction types including floating point operations, non-uniform address calculations, transcendental operations, integer operations, shifts, conditional evaluation, etc.
     - Instructions
   * - VMEM Instructions
     - The total number of vector memory operations issued.  These include most loads, stores and atomic operations and all accesses to [generic, global, private and texture](Mspace) memory.
     - Instructions
   * - [LDS](lds) Instructions
     - The total number of LDS (a.k.a., "shared memory") operations issued.  These include (e.g.,) loads, stores, atomics, and HIP's `__shfl` operations.
     - Instructions
   * - [MFMA](mfma) Instructions
     - The total number of matrix fused multiply-add instructions issued.
     - Instructions
   * - [SALU](salu) Instructions
     - The total number of scalar arithmetic logic unit (SALU) operations issued. Typically these are used for (e.g.,) address calculations, literal constants, and other operations that are _provably_ uniform across a wavefront.  Although scalar memory (SMEM) operations are issued by the SALU, they are counted separately in this section.
     - Instructions
   * - SMEM Instructions
     - The total number of scalar memory (SMEM) operations issued.  These are typically used for loading kernel arguments, base-pointers and loads from HIP's `__constant__` memory.
     - Instructions
   * - [Branch](branch) Instructions
     - The total number of branch operations issued.  These typically consist of jump / branch operations and are used to implement control flow.
     - Instructions

.. code:: {note}

   Note, as mentioned in the [Branch](branch) section: branch operations are not used for execution mask updates, but only for "whole wavefront" control-flow changes.

VALU Arithmetic Instruction Mix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: {warning}

   Not all metrics in this section (e.g., the floating-point instruction breakdowns) are available on CDNA accelerators older than the [MI2XX](2xxnote) series.

This panel details the various types of vector instructions that were
issued to the `VALU <valu>`__. The metrics in this section do *not*
include `MFMA <mfma>`__ instructions using the same precision, e.g. the
“F16-ADD” metric does not include any 16-bit floating point additions
executed as part of an MFMA instruction using the same precision.

.. code:: {list-table}

   :header-rows: 1
   :widths: 15 65 20
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - INT32
     - The total number of instructions operating on 32-bit integer operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - INT64
     - The total number of instructions operating on 64-bit integer operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F16-ADD
     - The total number of addition instructions operating on 16-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F16-MUL
     - The total number of multiplication instructions operating on 16-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F16-FMA
     - The total number of fused multiply-add instructions operating on 16-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F16-TRANS
     - The total number of transcendental instructions (e.g., `sqrt`) operating on 16-bit floating-point operands issued to the VALU per [normalization-unit](normunit)
     - Instructions per [normalization-unit](normunit)
   * - F32-ADD
     - The total number of addition instructions operating on 32-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F32-MUL
     - The total number of multiplication instructions operating on 32-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F32-FMA
     - The total number of fused multiply-add instructions operating on 32-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F32-TRANS
     - The total number of transcendental instructions (e.g., `sqrt`) operating on 32-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F64-ADD
     - The total number of addition instructions operating on 64-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F64-MUL
     - The total number of multiplication instructions operating on 64-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F64-FMA
     - The total number of fused multiply-add instructions operating on 64-bit floating-point operands issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - F64-TRANS
     - The total number of transcendental instructions (e.g., `sqrt`) operating on 64-bit floating-point operands issued to the VALUper [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Conversion
     - The total number of type conversion instructions (e.g., converting data to/from F32↔F64) issued to the VALU per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)

For an example of these counters in action, the reader is referred to
the `VALU Arithmetic Instruction Mix example <VALU_inst_mix_example>`__.

VMEM Instruction Mix
^^^^^^^^^^^^^^^^^^^^

This section breaks down the types of vector memory (VMEM) instructions
that were issued. Refer to the `Instruction Counts metrics
section <TA_inst>`__ of address-processor frontend of the vL1D cache for
a description of these VMEM instructions.

(MFMA_Inst_mix)= ##### MFMA Instruction Mix

.. code:: {warning}

   The metrics in this section are only available on CDNA2 ([MI2XX](2xxnote)) accelerators and newer.

This section details the types of Matrix Fused Multiply-Add
(`MFMA <mfma>`__) instructions that were issued. Note that
`MFMA <mfma>`__ instructions are classified by the type of input data
they operate on, and *not* the data-type the result is accumulated to.

.. code:: {list-table}

   :header-rows: 1
   :widths: 25 60 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - MFMA-I8 Instructions
     - The total number of 8-bit integer [MFMA](mfma) instructions issued per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - MFMA-F16 Instructions
     - The total number of 16-bit floating point [MFMA](mfma) instructions issued per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - MFMA-BF16 Instructions
     - The total number of 16-bit brain floating point [MFMA](mfma) instructions issued per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - MFMA-F32 Instructions
     - The total number of 32-bit floating-point [MFMA](mfma) instructions issued per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - MFMA-F64 Instructions
     - The total number of 64-bit floating-point [MFMA](mfma) instructions issued per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)

Compute Pipeline
----------------

FLOP counting conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

Omniperf’s conventions for VALU FLOP counting are as follows: - Addition
or Multiplication: 1 operation - Transcendentals: 1 operation - Fused
Multiply-Add (FMA): 2 operations

Integer operations (IOPs) do not use this convention. They are counted
as a single operation regardless of the instruction type.

.. code:: {note}

   Packed operations which operate on multiple operands in the same instruction are counted identically to the underlying instruction type.
   For example, the `v_pk_add_f32` instruction on [MI2XX](2xxnote), which performs an add operation on two pairs of aligned 32-bit floating-point operands is counted only as a single addition (i.e., 1 operation).

As discussed in the `Instruction Mix <Inst_Mix>`__ section, the FLOP/IOP
metrics in this section do not take into account the execution mask of
the operation, and will report the same value even if the execution mask
is identically zero.

For example, a FMA instruction operating on 32-bit floating-point
operands (e.g., ``v_fma_f32`` on a `MI2XX <2xxnote>`__ accelerator)
would be counted as 128 total FLOPs: 2 operations (due to the
instruction type) multiplied by 64 operations (because the wavefront is
composed of 64 work-items).

(Compute_SOL)= ##### Compute Speed-of-Light

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

This section reports the number of floating-point and integer operations
executed on the `VALU <valu>`__ and `MFMA <mfma>`__ units in various
precisions. We note that unlike the `VALU instruction
mix <VALU_Inst_Mix>`__ and `MFMA instruction mix <MFMA_Inst_mix>`__
sections, the metrics here are reported as FLOPs and IOPs, i.e., the
total number of operations executed.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - VALU FLOPs
     - The total floating-point operations executed per second on the [VALU](valu).  This is also presented as a percent of the peak theoretical FLOPs achievable on the specific accelerator. Note: this does not include any floating-point operations from [MFMA](mfma) instructions.
     - GFLOPs
   * - VALU IOPs
     - The total integer operations executed per second on the [VALU](valu).  This is also presented as a percent of the peak theoretical IOPs achievable on the specific accelerator. Note: this does not include any integer operations from [MFMA](mfma) instructions.
     - GIOPs
   * - MFMA FLOPs (BF16)
     - The total number of 16-bit brain floating point [MFMA](mfma) operations executed per second. Note: this does not include any 16-bit brain floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical BF16 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - MFMA FLOPs (F16)
     - The total number of 16-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 16-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F16 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - MFMA FLOPs (F32)
     - The total number of 32-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 32-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F32 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - MFMA FLOPs (F64)
     - The total number of 64-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 64-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F64 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - MFMA IOPs (INT8)
     - The total number of 8-bit integer [MFMA](mfma) operations executed per second. Note: this does not include any 8-bit integer operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical INT8 MFMA operations achievable on the specific accelerator.
     - GIOPs

(Pipeline_stats)= ##### Pipeline Statistics

This section reports a number of key performance characteristics of
various execution units on the `CU <cu>`__. The reader is referred to
the `Instructions per-cycle and Utilizations <IPC_example>`__ example
for a detailed dive into these metrics, and the
`scheduler <scheduler>`__ for a high-level overview of execution units
and instruction issue.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - IPC
     - The ratio of the total number of instructions executed on the [CU](cu) over the [total active CU cycles](TotalActiveCUCycles).
     - Instructions per-cycle
   * - IPC (Issued)
     - The ratio of the total number of (non-[internal](Internal_ipc)) instructions issued over the number of cycles where the [scheduler](scheduler) was actively working on issuing instructions.  The reader is recommended the [Issued IPC](Issued_ipc) example for further detail.
     - Instructions per-cycle
   * - SALU Utilization
     - Indicates what percent of the kernel's duration the [SALU](salu) was busy executing instructions.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [SALU](salu) / [SMEM](salu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - VALU Utilization
     - Indicates what percent of the kernel's duration the [VALU](valu) was busy executing instructions.  Does not include [VMEM](valu) operations.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [VALU](valu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - VMEM Utilization
     - Indicates what percent of the kernel's duration the [VMEM](valu) unit was busy executing instructions, including both global/generic and spill/scratch operations (see the [VMEM instruction count metrics](TA_inst) for more detail).  Does not include [VALU](valu) operations.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [VMEM](valu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - Branch Utilization
     - Indicates what percent of the kernel's duration the [Branch](branch) unit was busy executing instructions. Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [Branch](branch) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - VALU Active Threads
     - Indicates the average level of [divergence](Divergence) within a wavefront over the lifetime of the kernel. The number of work-items that were active in a wavefront during execution of each [VALU](valu) instruction, time-averaged over all VALU instructions run on all wavefronts in the kernel.
     - Work-items
   * - MFMA Utilization
     - Indicates what percent of the kernel's duration the [MFMA](mfma) unit was busy executing instructions.  Computed as the ratio of the total number of cycles spent by the [MFMA](salu) was busy over the [total CU cycles](TotalCUCycles).
     - Percent
   * - MFMA Instruction Cycles
     - The average duration of [MFMA](mfma) instructions in this kernel in cycles.  Computed as the ratio of the total number of cycles the [MFMA](mfma) unit was busy over the total number of [MFMA](mfma) instructions.  Compare to e.g., the [AMD Matrix Instruction Calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator).
     - Cycles per instruction
   * - VMEM Latency
     - The average number of round-trip cycles (i.e., from issue to data-return / acknowledgment) required for a VMEM instruction to complete.
     - Cycles
   * - SMEM Latency
     - The average number of round-trip cycles (i.e., from issue to data-return / acknowledgment) required for a SMEM instruction to complete.
     - Cycles

.. code:: {note}

   The Branch utilization reported in this section also includes time spent in other instruction types (namely: `s_endpgm`) that are _typically_ a very small percentage of the overall kernel execution.  This complication is omitted for simplicity, but may result in small amounts of "branch" utilization (<<1\%) for otherwise branch-less kernels.

(FLOPS)= ##### Arithmetic Operations

This section reports the total number of floating-point and integer
operations executed in various precisions. Unlike the `Compute
speed-of-light <Compute_SOL>`__ panel, this section reports both
`VALU <valu>`__ and `MFMA <mfma>`__ operations of the same precision
(e.g., F32) in the same metric. Additionally, this panel lets the user
control how the data is normalized (i.e., control the
`normalization-unit <normunit>`__), while the speed-of-light panel does
not. For more detail on how operations are counted see the `FLOP
counting convention <FLOP_count>`__ section.

.. code:: {warning}

   As discussed in the [Instruction Mix](Inst_Mix) section, the metrics in this section do not take into account the execution mask of the operation, and will report the same value even if EXEC is identically zero.

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - FLOPs (Total)
     - The total number of floating-point operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit)
     - FLOP per [normalization-unit](normunit)
   * - IOPs (Total)
     - The total number of integer operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit)
     - IOP per [normalization-unit](normunit)
   * - F16 OPs
     - The total number of 16-bit floating-point operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit)
     - FLOP per [normalization-unit](normunit)
   * - BF16 OPs
     - The total number of 16-bit brain floating-point operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit). Note: on current CDNA accelerators, the [VALU](valu) has no native BF16 instructions.
     - FLOP per [normalization-unit](normunit)
   * - F32 OPs
     - The total number of 32-bit floating-point operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit)
     - FLOP per [normalization-unit](normunit)
   * - F64 OPs
     - The total number of 64-bit floating-point operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit)
     - FLOP per [normalization-unit](normunit)
   * - INT8 OPs
     - The total number of 8-bit integer operations executed on either the [VALU](valu) or [MFMA](mfma) units, per [normalization-unit](normunit). Note: on current CDNA accelerators, the [VALU](valu) has no native INT8 instructions.
     - IOPs per [normalization-unit](normunit)

(LDS_metrics)= ### Local Data Share (LDS)

LDS Speed-of-Light
^^^^^^^^^^^^^^^^^^

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

The LDS speed-of-light chart shows a number of key metrics for the
`LDS <lds>`__ as a comparison with the peak achievable values of those
metrics. The reader is referred to our previous `LDS <lds>`__
description for a more in-depth view of the hardware.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Utilization
     - Indicates what percent of the kernel's duration the [LDS](lds) was actively executing instructions (including, but not limited to, load, store, atomic and HIP's `__shfl` operations).  Calculated as the ratio of the total number of cycles LDS was active over the [total CU cycles](TotalCUCycles).
     - Percent
   * - Access Rate
     - Indicates the percentage of SIMDs in the [VALU](valu){sup}`1` actively issuing LDS instructions, averaged over the lifetime of the kernel. Calculated as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [LDS](lds) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - Theoretical Bandwidth (% of Peak)
     - Indicates the maximum amount of bytes that _could_ have been loaded from/stored to/atomically updated in the LDS in this kernel, as a percent of the peak LDS bandwidth achievable.  See the [LDS Bandwidth example](lds_bandwidth) for more detail.
     - Percent
   * - Bank Conflict Rate
     - Indicates the percentage of active LDS cycles that were spent servicing bank conflicts. Calculated as the ratio of LDS cycles spent servicing bank conflicts over the number of LDS cycles that would have been required to move the same amount of data in an uncontended access.{sup}`2`
     - Percent

.. code:: {note}

   {sup}`1` Here we assume the typical case where the workload evenly distributes LDS operations over all SIMDs in a CU (that is, waves on different SIMDs are executing similar code).
   For highly unbalanced workloads, where e.g., one SIMD pair in the CU does not issue LDS instructions at all, this metric is better interpreted as the percentage of SIMDs issuing LDS instructions on [SIMD pairs](lds) that are actively using the LDS, averaged over the lifetime of the kernel.

   {sup}`2` The maximum value of the bank conflict rate is less than 100% (specifically: 96.875%), as the first cycle in the [LDS scheduler](lds) is never considered contended.

Statistics
^^^^^^^^^^

The `LDS <lds>`__ statistics panel gives a more detailed view of the
hardware:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - LDS Instructions
     - The total number of LDS instructions (including, but not limited to, read/write/atomics, and e.g., HIP's `__shfl` instructions) executed per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Theoretical Bandwidth
     - Indicates the maximum amount of bytes that could have been loaded from/stored to/atomically updated in the LDS per [normalization-unit](normunit).  Does _not_ take into account the execution mask of the wavefront when the instruction was executed (see [LDS Bandwidth](lds_bandwidth) example for more detail).
     - Bytes per [normalization-unit](normunit)
   * - LDS Latency
     - The average number of round-trip cycles (i.e., from issue to data-return / acknowledgment) required for an LDS instruction to complete.
     - Cycles
   * - Bank Conflicts/Access
     - The ratio of the number of cycles spent in the [LDS scheduler](lds) due to bank conflicts (as determined by the conflict resolution hardware) to the base number of cycles that would be spent in the LDS scheduler in a completely uncontended case.  This is the unnormalized form of the Bank Conflict Rate.
     - Conflicts/Access
   * - Index Accesses
     - The total number of cycles spent in the [LDS scheduler](lds) over all operations per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Atomic Return Cycles
     - The total number of cycles spent on LDS atomics with return  per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Bank Conflicts
     - The total number of cycles spent in the [LDS scheduler](lds) due to bank conflicts (as determined by the conflict resolution hardware) per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Address Conflicts
     - The total number of cycles spent in the [LDS scheduler](lds) due to address conflicts (as determined by the conflict resolution hardware) per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Unaligned Stall
     - The total number of cycles spent in the [LDS scheduler](lds) due to stalls from non-dword aligned addresses per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Memory Violations
     - The total number of out-of-bounds accesses made to the LDS, per [normalization-unit](normunit).  This is unused and expected to be zero in most configurations for modern CDNA accelerators.
     - Accesses per [normalization-unit](normunit)

(vL1D)= ### Vector L1 Cache (vL1D)

The vector L1 data (vL1D) cache is local to each `compute unit <CU>`__
on the accelerator, and handles vector memory operations issued by a
wavefront. The vL1D cache consists of several components:

-  an address processing unit, also known as the `texture addresser
   (TA) <TA>`__, which receives commands (e.g., instructions) and
   write/atomic data from the `Compute Unit <CU>`__, and coalesces them
   into fewer requests for the cache to process.
-  an address translation unit, also known as the L1 Unified Translation
   Cache (UTCL1), that translates requests from virtual to physical
   addresses for lookup in the cache. The translation unit has an L1
   translation lookaside buffer (L1TLB) to reduce the cost of repeated
   translations.
-  a Tag RAM that looks up whether a requested cache line is already
   present in the `cache <TC>`__.
-  the result of the Tag RAM lookup is placed in the L1 cache controller
   for routing to the correct location, e.g., the `L2 Memory
   Interface <TCP_TCC_Transactions_Detail>`__ for misses or the `Cache
   RAM <TC>`__ for hits.
-  the Cache RAM, also known as the `texture cache (TC) <TC>`__, stores
   requested data for potential reuse. Data returned from the `L2
   cache <L2>`__ is placed into the Cache RAM before going down the
   `data-return path <TD>`__.
-  a backend data processing unit, also known as the `texture data
   (TD) <TD>`__ that routes data back to the requesting `Compute
   Unit <CU>`__.

Together, this complex is known as the vL1D, or Texture Cache per Pipe
(TCP). A simplified diagram of the vL1D is presented below:

\```{figure} images/l1perf_model.\* :scale: 150 % :alt: Performance
model of the vL1D Cache on AMD Instinct(tm) MI accelerators. :align:
center

Performance model of the vL1D Cache on AMD Instinct(tm) MI accelerators.

::


   (L1_SOL)=
   #### vL1D Speed-of-Light

   ```{warning}
   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

The vL1D’s speed-of-light chart shows several key metrics for the vL1D
as a comparison with the peak achievable values of those metrics.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Hit Rate
     - The ratio of the number of vL1D cache line requests that hit{sup}`1` in vL1D cache over the total number of cache line requests to the [vL1D Cache RAM](TC).
     - Percent
   * - Bandwidth
     - The number of bytes looked up in the vL1D cache as a result of [VMEM](VALU) instructions, as a percent of the peak theoretical bandwidth achievable on the specific accelerator.  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.
     - Percent
   * - Utilization
     - Indicates how busy the [vL1D Cache RAM](TC) was during the kernel execution. The number of cycles where the [vL1D Cache RAM](TC) is actively processing any request divided by the number of cycles where the [vL1D is active](vL1d_activity){sup}`2`
     - Percent
   * - Coalescing
     - Indicates how well memory instructions were coalesced by the [address processing unit](TA), ranging from uncoalesced (25\%) to fully coalesced (100\%). The average number of [thread-requests](ThreadRequests) generated per instruction divided by the ideal number of [thread-requests](ThreadRequests) per instruction.
     - Percent

(vL1d_activity)=

.. code:: {note}

   {sup}`1` The vL1D cache on AMD Instinct(tm) MI CDNA accelerators uses a "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'hit'.
   Therefore, it is also important to consider the Access Latency metric in the [Cache access metrics](TCP_cache_access_metrics) section when evaluating the vL1D hit rate.

   {sup}`2` Omniperf considers the vL1D to be active when any part of the vL1D (excluding the [address-processor](TA) and [data-return](TD) units) are active, e.g., performing a translation, waiting for data, accessing the Tag or Cache RAMs, etc.

(TA)= #### Address Processing Unit or Texture Addresser (TA)

The `vL1D <vL1D>`__\ ’s address processing unit receives vector memory
instructions (commands) along with write/atomic data from a `Compute
Unit <CU>`__ and is responsible for coalescing these into requests for
lookup in the `vL1D RAM <TC>`__. The address processor passes
information about the commands (coalescing state, destination SIMD,
etc.) to the `data processing unit <TD>`__ for use after the requested
data has been retrieved.

Omniperf reports several metrics to indicate performance bottlenecks in
the address processing unit, which are broken down into a few
categories:

-  Busy / stall metrics
-  Instruction counts
-  Spill / Stack metrics

Busy / Stall metrics
''''''''''''''''''''

When executing vector memory instructions, the compute unit must send an
address (and in the case of writes/atomics, data) to the address
processing unit. When the frontend cannot accept any more addresses, it
must backpressure the wave-issue logic for the VMEM pipe and prevent the
issue of a vector memory instruction until a previously issued memory
operation has been processed.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Busy
     - Percent of the [total CU cycles](TotalCUCycles) the address processor was busy
     - Percent
   * - Address Stall
     - Percent of the [total CU cycles](TotalCUCycles) the address processor was stalled from sending address requests further into the vL1D pipeline
     - Percent
   * - Data Stall
     - Percent of the [total CU cycles](TotalCUCycles) the address processor was stalled from sending write/atomic data further into the vL1D pipeline
     - Percent
   * - Data-Processor → Address Stall
     - Percent of [total CU cycles](TotalCUCycles) the address processor was stalled waiting to send command data to the [data processor](TD)
     - Percent

(TA_inst)= ##### Instruction counts

The address processor also counts instruction types to give the user
information on what sorts of memory instructions were executed by the
kernel. These are broken down into a few major categories:

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 20 60
   :class: noscroll-table
   * - Memory type
     - Usage
     - Description
   * - Global
     - Global memory
     - Global memory can be seen by all threads from a process.  This includes the local accelerator's DRAM, remote accelerator's DRAM, and the host's DRAM.
   * - Generic
     - Dynamic address spaces
     - Generic memory, a.k.a. "flat" memory, is used when the compiler cannot statically prove that a pointer is to memory in one or the other address spaces. The pointer could dynamically point into global, local, constant, or private memory.
   * - Private Memory
     - Register spills / Stack memory
     - Private memory, a.k.a. "scratch" memory, is only visible to a particular [work-item](workitem) in a particular [workgroup](workgroup).  On AMD Instinct(tm) MI accelerators, private memory is used to implement both register spills and stack memory accesses.

The address processor counts these instruction types as follows:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table

   * - Type
     - Description
     - Unit
   * - Global/Generic
     - The total number of global & generic memory instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Global/Generic Read
     - The total number of global & generic memory read instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Global/Generic Write
     - The total number of global & generic memory write instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Global/Generic Atomic
     - The total number of global & generic memory atomic (with and without return) instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Spill/Stack
     - The total number of spill/stack memory instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Spill/Stack Read
     - The total number of spill/stack memory read instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Spill/Stack Write
     - The total number of spill/stack memory write instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit).
     - Instruction per [normalization-unit](normunit)
   * - Spill/Stack Atomic
     - The total number of spill/stack memory atomic (with and without return) instructions executed on all [compute units](CU) on the accelerator, per [normalization-unit](normunit). Typically unused as these memory operations are typically used to implement thread-local storage.
     - Instructions per [normalization-unit](normunit)

.. code:: {note}

   The above is a simplified model specifically for the HIP programming language that does not consider (e.g.,) inline assembly usage, constant memory usage or texture memory.

   These categories correspond to:
     - Global/Generic: global and flat memory operations, that are used for Global and Generic memory access.
     - Spill/Stack: buffer instructions which are used on the MI50, MI100, and [MI2XX](2xxnote) accelerators for register spills / stack memory.

   These concepts are described in more detail in the [memory space section](Mspace) below, while generic memory access is explored in the [generic memory benchmark](flatmembench) section.

Spill/Stack metrics
'''''''''''''''''''

Finally, the address processing unit contains a separate coalescing
stage for spill/stack memory, and thus reports:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Spill/Stack Total Cycles
     - The number of cycles the address processing unit spent working on spill/stack instructions, per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Spill/Stack Coalesced Read Cycles
     - The number of cycles the address processing unit spent working on coalesced spill/stack read instructions, per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)
   * - Spill/Stack Coalesced Write Cycles
     - The number of cycles the address processing unit spent working on coalesced spill/stack write instructions, per [normalization-unit](normunit)
     - Cycles per [normalization-unit](normunit)

(UTCL1)= #### L1 Unified Translation Cache (UTCL1)

After a vector memory instruction has been processed/coalesced by the
address processing unit of the vL1D, it must be translated from a
virtual to physical address. This process is handled by the L1 Unified
Translation Cache (UTCL1). This cache contains a L1 Translation
Lookaside Buffer (TLB) which stores recently translated addresses to
reduce the cost of subsequent re-translations.

Omniperf reports the following L1 TLB metrics:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Requests
     - The number of translation requests made to the UTCL1 per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Hits
     - The number of translation requests that hit in the UTCL1, and could be reused, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Hit Ratio
     - The ratio of the number of translation requests that hit in the UTCL1 divided by the total number of translation requests made to the UTCL1.
     - Percent
   * - Translation Misses
     - The total number of translation requests that missed in the UTCL1 due to translation not being present in the cache, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Permission Misses
     - The total number of translation requests that missed in the UTCL1 due to a permission error, per [normalization-unit](normunit).  This is unused and expected to be zero in most configurations for modern CDNA accelerators.
     - Requests per [normalization-unit](normunit)

.. code:: {note}

   On current CDNA accelerators, such as the [MI2XX](2xxnote), the UTCL1 does _not_ count hit-on-miss requests.

(TC)= #### Vector L1 Cache RAM (TC)

After coalescing in the `address processing unit <TA>`__ of the v1LD,
and address translation in the `L1 TLB <UTCL1>`__ the request proceeds
to the Cache RAM stage of the pipeline. Incoming requests are looked up
in the cache RAMs using parts of the physical address as a tag. Hits
will be returned through the `data-return path <TD>`__, while misses
will routed out to the `L2 Cache <L2>`__ for servicing.

The metrics tracked by the vL1D RAM include:

-  Stall metrics
-  Cache access metrics
-  vL1D-L2 transaction detail metrics

(TCP_cache_stall_metrics)= ##### vL1D cache stall metrics

The vL1D also reports where it is stalled in the pipeline, which may
indicate performance limiters of the cache. A stall in the pipeline may
result in backpressuring earlier parts of the pipeline, e.g., a stall on
L2 requests may backpressure the wave-issue logic of the `VMEM <VALU>`__
pipe and prevent it from issuing more vector memory instructions until
the vL1D’s outstanding requests are completed.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Stalled on L2 Data
     - The ratio of the number of cycles where the vL1D is stalled waiting for requested data to return from the [L2 cache](L2) divided by the number of cycles where the [vL1D is active](vL1d_activity).
     - Percent
   * - Stalled on L2 Requests
     - The ratio of the number of cycles where the vL1D is stalled waiting to issue a request for data to the [L2 cache](L2) divided by the number of cycles where the [vL1D is active](vL1d_activity).
     - Percent
   * - Tag RAM Stall (Read/Write/Atomic)
     - The ratio of the number of cycles where the vL1D is stalled due to Read/Write/Atomic requests with conflicting tags being looked up concurrently, divided by the number of cycles where the [vL1D is active](vL1d_activity).
     - Percent

(TCP_cache_access_metrics)= ##### vL1D cache access metrics

The vL1D cache access metrics broadly indicate the type of requests
incoming from the `cache frontend <TA>`__, the number of requests that
were serviced by the vL1D, and the number & type of outgoing requests to
the `L2 cache <L2>`__. In addition, this section includes the
approximate latencies of accesses to the cache itself, along with
latencies of read/write memory operations to the `L2 cache <L2>`__.

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Total Requests
     - The total number of incoming requests from the [address processing unit](TA) after coalescing.
     - Requests
   * - Total read/write/atomic requests
     - The total number of incoming read/write/atomic requests from the [address processing unit](TA) after coalescing per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Cache Bandwidth
     - The number of bytes looked up in the vL1D cache as a result of [VMEM](VALU) instructions per [normalization-unit](normunit).  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.
     - Bytes per [normalization-unit](normunit)
   * - Cache Hit Rate
     - The ratio of the number of vL1D cache line requests that hit in vL1D cache over the total number of cache line requests to the [vL1D Cache RAM](TC).
     - Percent
   * - Cache Accesses
     - The total number of cache line lookups in the vL1D.
     - Cache lines
   * - Cache Hits
     - The number of cache accesses minus the number of outgoing requests to the [L2 cache](L2), i.e., the number of cache line requests serviced by the [vL1D Cache RAM](TC) per [normalization-unit](normunit).
     - Cache lines per [normalization-unit](normunit)
   * - Invalidations
     - The number of times the vL1D was issued a write-back invalidate command during the kernel's execution per [normalization-unit](normunit).  This may be triggered by, e.g., the `buffer_wbinvl1` instruction.
     - Invalidations per [normalization-unit](normunit)
   * - L1-L2 Bandwidth
     - The number of bytes transferred across the vL1D-L2 interface as a result of [VMEM](VALU) instructions, per [normalization-unit](normunit).  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.
     - Bytes per [normalization-unit](normunit)
   * - L1-L2 Reads
     - The number of read requests for a vL1D cache line that were not satisfied by the vL1D and must be retrieved from the to the [L2 Cache](L2) per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - L1-L2 Writes
     - The number of post-coalescing write requests that are sent through the vL1D to the [L2 cache](L2), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - L1-L2 Atomics
     - The number of atomic requests that are sent through the vL1D to the [L2 cache](L2), per [normalization-unit](normunit). This includes requests for atomics with, and without return.
     - Requests per [normalization-unit](normunit)
   * - L1 Access Latency
     - The average number of cycles that a vL1D cache line request spent in the vL1D cache pipeline.
     - Cycles
   * - L1-L2 Read Access Latency
     - The average number of cycles that the vL1D cache took to issue and receive read requests from the [L2 Cache](L2).  This number also includes requests for atomics with return values.
     - Cycles
   * - L1-L2 Write Access Latency
     - The average number of cycles that the vL1D cache took to issue and receive acknowledgement of a write request to the [L2 Cache](L2).  This number also includes requests for atomics without return values.
     - Cycles

.. code:: {note}

   All cache accesses in vL1D are for a single cache line's worth of data.
   The size of a cache line may vary, however on current AMD Instinct(tm) MI CDNA accelerators and GCN GPUs the L1 cache line size is 64B.

(TCP_TCC_Transactions_Detail)= ##### vL1D - L2 Transaction Detail

This section provides a more granular look at the types of requests made
to the `L2 cache <L2>`__. These are broken down by the operation type
(read / write / atomic, with, or without return), and the `memory
type <Mtype>`__. For more detail, the reader is referred to the `Memory
Types <Mtype>`__ section.

(TD)= #### Vector L1 Data-Return Path or Texture Data (TD)

The data-return path of the vL1D cache, also known as the Texture Data
(TD) unit, is responsible for routing data returned from the `vL1D cache
RAM <TC>`__ back to a wavefront on a SIMD. As described in the `vL1D
cache front-end <TA>`__ section, the data-return path is passed
information about the space requirements and routing for data requests
from the `VALU <valu>`__. When data is returned from the `vL1D cache
RAM <TC>`__, it is matched to this previously stored request data, and
returned to the appropriate SIMD.

Omniperf reports the following vL1D data-return path metrics:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Data-return Busy
     - Percent of the [total CU cycles](TotalCUCycles) the data-return unit was busy processing or waiting on data to return to the [CU](CU).
     - Percent
   * - Cache RAM → Data-return Stall
     - Percent of the [total CU cycles](TotalCUCycles) the data-return unit was stalled on data to be returned from the [vL1D Cache RAM](TC).
     - Percent
   * - Workgroup manager → Data-return Stall
     - Percent of the [total CU cycles](TotalCUCycles) the data-return unit was stalled by the [workgroup manager](SPI) due to initialization of registers as a part of launching new workgroups.
     - Percent
   * - Coalescable Instructions
     - The number of instructions submitted to the [data-return unit](TD) by the [address-processor](TA) that were found to be coalescable, per [normalization-unit](normunit).
     - Instructions per [normalization-unit](normunit)
   * - Read Instructions
     - The number of read instructions submitted to the [data-return unit](TD) by the [address-processor](TA) summed over all [compute units](CU) on the accelerator, per [normalization-unit](normunit).  This is expected to be the sum of global/generic and spill/stack reads in the [address processor](TA_inst).
     - Instructions per [normalization-unit](normunit)
   * - Write Instructions
     - The number of store instructions submitted to the [data-return unit](TD) by the [address-processor](TA) summed over all [compute units](CU) on the accelerator, per [normalization-unit](normunit).  This is expected to be the sum of global/generic and spill/stack stores counted by the [vL1D cache-frontend](TA_inst).
     - Instructions per [normalization-unit](normunit)
   * - Atomic Instructions
     - The number of atomic instructions submitted to the [data-return unit](TD) by the [address-processor](TA) summed over all [compute units](CU) on the accelerator, per [normalization-unit](normunit).  This is expected to be the sum of global/generic and spill/stack atomics in the [address processor](TA_inst).
     - Instructions per [normalization-unit](normunit)

(L2)= ## L2 Cache (TCC)

The L2 cache is the coherence point for current AMD Instinct(tm) MI GCN
GPUs and CDNA accelerators, and is shared by all `compute units <CU>`__
on the device. Besides serving requests from the `vector L1 data
caches <vL1D>`__, the L2 cache also is responsible for servicing
requests from the `L1 instruction caches <L1I>`__, the `scalar L1 data
caches <sL1D>`__ and the `command-processor <CP>`__. The L2 cache is
composed of a number of distinct channels (32 on
MI100/`MI2XX <2xxnote>`__ series CDNA accelerators at 256B address
interleaving) which can largely operate independently. Mapping of
incoming requests to a specific L2 channel is determined by a hashing
mechanism that attempts to evenly distribute requests across the L2
channels. Requests that miss in the L2 cache are passed out to `Infinity
Fabric(tm) <l2fabric>`__ to be routed to the appropriate memory
location.

The L2 cache metrics reported by Omniperf are broken down into four
categories:

-  L2 Speed-of-Light
-  L2 Cache Accesses
-  L2-Fabric Transactions
-  L2-Fabric Stalls

(L2SoL)= ### L2 Speed-of-Light

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

The L2 cache’s speed-of-light table contains a few key metrics about the
performance of the L2 cache, aggregated over all the L2 channels, as a
comparison with the peak achievable values of those metrics:

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Utilization
     - The ratio of the [number of cycles an L2 channel was active, summed over all L2 channels on the accelerator](TotalActiveL2Cycles) over the [total L2 cycles](TotalL2Cycles).
     - Percent
   * - Bandwidth
     - The number of bytes looked up in the L2 cache, as a percent of the peak theoretical bandwidth achievable on the specific accelerator.  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.
     - Percent
   * - Hit Rate
     - The ratio of the number of L2 cache line requests that hit in the L2 cache over the total number of incoming cache line requests to the L2 cache.
     - Percent
   * - L2-Fabric Read BW
     - The number of bytes read by the L2 over the [Infinity Fabric(tm) interface](l2fabric) per unit time.
     - GB/s
   * - L2-Fabric Write and Atomic BW
     - The number of bytes sent by the L2 over the [Infinity Fabric(tm) interface](l2fabric) by write and atomic operations per unit time.
     - GB/s

.. code:: {note}

   The L2 cache on AMD Instinct(tm) MI CDNA accelerators uses a "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'hit'.
   Therefore, it is also important to consider the latency metric in the [L2-Fabric](l2fabric) section when evaluating the L2 hit rate.

(L2_cache_metrics)= ### L2 Cache Accesses

This section details the incoming requests to the L2 cache from the
`vL1D <vL1D>`__ and other clients (e.g., the `sL1D <sL1D>`__ and
`L1I <L1I>`__ caches).

.. code:: {list-table}

   :header-rows: 1
   :widths: 13 70 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Bandwidth
     - The number of bytes looked up in the L2 cache, per [normalization-unit](normunit).  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.
     - Bytes per [normalization-unit](normunit)
   * - Requests
     - The total number of incoming requests to the L2 from all clients for all request types, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Read Requests
     - The total number of read requests to the L2 from all clients.
     - Requests per [normalization-unit](normunit)
   * - Write Requests
     - The total number of write requests to the L2 from all clients.
     - Requests per [normalization-unit](normunit)
   * - Atomic Requests
     - The total number of atomic requests (with and without return) to the L2 from all clients.
     - Requests per [normalization-unit](normunit)
   * - Streaming Requests
     - The total number of incoming requests to the L2 that are marked as 'streaming'. The exact meaning of this may differ depending on the targeted accelerator, however on an [MI2XX](2xxnote) this corresponds to [non-temporal load or stores](https://clang.llvm.org/docs/LanguageExtensions.html#non-temporal-load-store-builtins). The L2 cache attempts to evict 'streaming' requests before normal requests when the L2 is at capacity.
     - Requests per [normalization-unit](normunit)
   * - Probe Requests
     - The number of coherence probe requests made to the L2 cache from outside the accelerator.  On an [MI2XX](2xxnote), probe requests may be generated by e.g., writes to [fine-grained device](MType) memory or by writes to [coarse-grained](MType) device memory.
     - Requests per [normalization-unit](normunit)
   * - Hit Rate
     - The ratio of the number of L2 cache line requests that hit in the L2 cache over the total number of incoming cache line requests to the L2 cache.
     - Percent
   * - Hits
     - The total number of requests to the L2 from all clients that hit in the cache.  As noted in the [speed-of-light](L2SoL) section, this includes hit-on-miss requests.
     - Requests per [normalization-unit](normunit)
   * - Misses
     - The total number of requests to the L2 from all clients that miss in the cache.  As noted in the [speed-of-light](L2SoL) section, these do not include hit-on-miss requests.
     - Requests per [normalization-unit](normunit)
   * - Writebacks
     - The total number of L2 cache lines written back to memory for any reason. Write-backs may occur due to e.g., user-code (e.g., HIP kernel calls to `__threadfence_system`, or atomic built-ins), by the [command-processor](CP)'s memory acquire/release fences, or for other internal hardware reasons.
     - Cache lines per [normalization-unit](normunit)
   * - Writebacks (Internal)
     - The total number of L2 cache lines written back to memory for internal hardware reasons, per [normalization-unit](normunit).
     - Cache lines per [normalization-unit](normunit)
   * - Writebacks (vL1D Req)
     - The total number of L2 cache lines written back to memory due to requests initiated by the [vL1D cache](vL1D), per [normalization-unit](normunit).
     - Cache lines per [normalization-unit](normunit)
   * - Evictions (Normal)
     - The total number of L2 cache lines evicted from the cache due to capacity limits, per [normalization-unit](normunit), per [normalization-unit](normunit).
     - Cache lines per [normalization-unit](normunit)
   * - Evictions (vL1D Req)
     - The total number of L2 cache lines evicted from the cache due to invalidation requests initiated by the [vL1D cache](vL1D), per [normalization-unit](normunit).
     - Cache lines per [normalization-unit](normunit)
   * - Non-hardware-Coherent Requests
     - The total number of requests to the L2 to Not-hardware-Coherent (NC) memory allocations, per [normalization-unit](normunit).  See the [Memory Types section](Mtype) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Uncached Requests
     - The total number of requests to the L2 that to uncached (UC) memory allocations.  See the [Memory Types section](Mtype) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Coherently Cached Requests
     - The total number of requests to the L2 that to coherently cachable (CC) memory allocations.  See the [Memory Types section](Mtype) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Read/Write Coherent Requests
     - The total number of requests to the L2 that to Read-Write coherent memory (RW) allocations.  See the [Memory Types section](Mtype) for more detail.
     - Requests per [normalization-unit](normunit)

.. code:: {note}

   All requests to the L2 are for a single cache line's worth of data.
   The size of a cache line may vary depending on the accelerator, however on an AMD Instinct(tm) CDNA2 [MI2XX](2xxnote) accelerator, it is 128B, while on an MI100, it is 64B.

(l2fabric)= ### L2-Fabric transactions

Requests/data that miss in the L2 must be routed to memory in order to
service them. The backing memory for a request may be local to this
accelerator (i.e., in the local high-bandwidth memory), in a remote
accelerator’s memory, or even in the CPU’s memory. Infinity Fabric(tm)
is responsible for routing these memory requests/data to the correct
location and returning any fetched data to the L2 cache. The `following
section <L2_req_flow>`__ describes the flow of these requests through
Infinity Fabric(tm) in more detail, as described by Omniperf metrics,
while `later sections <L2_req_metrics>`__ give detailed definitions of
individual metrics.

(L2_req_flow)= #### Request flow

Below is a diagram that illustrates how L2↔Fabric requests are reported
by Omniperf:

\```{figure} images/fabric.png :alt: L2↔Fabric transaction flow on AMD
Instinct(tm) MI accelerators. :align: center :name: fabric-fig

L2↔Fabric transaction flow on AMD Instinct(tm) MI accelerators.

::


   Requests from the L2 Cache are broken down into two major categories, read requests and write requests (at this granularity, atomic requests are treated as writes).

   From there, these requests can additionally subdivided in a number of ways.
   First, these requests may be sent across Infinity Fabric(tm) as different transaction sizes, 32B or 64B on current CDNA accelerators.

   ```{note}
   On current CDNA accelerators, the 32B read request path is expected to be unused (hence: is disconnected in the flow diagram).

In addition, the read and write requests can be further categorized as:
- uncached read/write requests, e.g., for accesses to `fine-grained
memory <Mtype>`__ - atomic requests, e.g., for atomic updates to
`fine-grained memory <Mtype>`__ - HBM read/write requests OR remote
read/write requests, i.e., for requests to the accelerator’s local HBM
OR requests to a remote accelerator’s HBM / the CPU’s DRAM.

These classifications are not necessarily *exclusive*, for example, a
write request can be classified as an atomic request to the
accelerator’s local HBM, and an uncached write request. The request-flow
diagram marks *exclusive* classifications as a splitting of the flow,
while *non-exclusive* requests do not split the flow line. For example,
a request is either a 32B Write Request OR a 64B Write request, as the
flow splits at this point: \```{figure} images/split.\* :scale: 50 %
:alt: Request flow splitting :align: center :name:
split-request-flow-fig

Splitting request flow

::

   However, continuing along, the same request might be an Atomic request and an Uncached Write request, as reflected by a non-split flow:
   ```{figure} images/nosplit.*
   :scale: 50 %
   :alt: Request flow splitting
   :align: center
   :name: nosplit-request-flow-fig

   Non-splitting request flow

Finally, we note that `uncached <Mtype>`__ read requests (e.g., to
`fine-grained memory <Mtype>`__) are handled specially on CDNA
accelerators, as indicated in the request flow diagram. These are
expected to be counted as a 64B Read Request, and *if* they are requests
to uncached memory (denoted by the dashed line), they will also be
counted as *two* uncached read requests (i.e., the request is split):

\```{figure} images/uncached.\* :scale: 50 % :alt: Uncached read-request
splitting :align: center :name: uncached-read-request-flow-fig

Uncached read-request splitting.

::


   (L2_req_metrics)=
   #### Metrics


   The following metrics are reported for the L2-Fabric interface:

   ```{list-table}
   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - L2-Fabric Read Bandwidth
     - The total number of bytes read by the L2 cache from Infinity Fabric(tm) per [normalization-unit](normunit).
     - Bytes per [normalization-unit](normunit)
   * - HBM Read Traffic
     - The percent of read requests generated by the L2 cache that are routed to the accelerator's local high-bandwidth memory (HBM).  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric Read bandwidth directed to the local HBM.
     - Percent
   * - Remote Read Traffic
     - The percent of read requests generated by the L2 cache that are routed to any memory location other than the accelerator's local high-bandwidth memory (HBM) --- e.g., the CPU's DRAM, a remote accelerator's HBM, etc. This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric Read bandwidth directed to a remote location.
     - Percent
   * - Uncached Read Traffic
     - The percent of read requests generated by the L2 cache that are reading from an [uncached memory allocation](Mtype).  Note, as described in the [request-flow](L2_req_flow) section, a single 64B read request is typically counted as two uncached read requests, hence it is possible for the Uncached Read Traffic to reach up to 200% of the total number of read requests.  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric read bandwidth directed to an uncached memory location.
     - Percent
   * - L2-Fabric Write and Atomic Bandwidth
     - The total number of bytes written by the L2 over Infinity Fabric(tm) by write and atomic operations per [normalization-unit](normunit). Note that on current CDNA accelerators, such as the [MI2XX](2xxnote), requests are only considered 'atomic' by Infinity Fabric(tm) if they are targeted at non-write-cachable memory, e.g., [fine-grained memory](Mtype) allocations or [uncached memory](Mtype) allocations on the [MI2XX](2xxnote).
     - Bytes per [normalization-unit](normunit)
   * - HBM Write and Atomic Traffic
     - The percent of write and atomic requests generated by the L2 cache that are routed to the accelerator's local high-bandwidth memory (HBM).  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric Write and Atomic bandwidth directed to the local HBM. Note that on current CDNA accelerators, such as the [MI2XX](2xxnote), requests are only considered 'atomic' by Infinity Fabric(tm) if they are targeted at [fine-grained memory](Mtype) allocations or [uncached memory](Mtype) allocations.
     - Percent
   * - Remote Write and Atomic Traffic
     - The percent of write and atomic requests generated by the L2 cache that are routed to any memory location other than the accelerator's local high-bandwidth memory (HBM) --- e.g., the CPU's DRAM, a remote accelerator's HBM, etc.  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric Write and Atomic bandwidth directed to a remote location. Note that on current CDNA accelerators, such as the [MI2XX](2xxnote), requests are only considered 'atomic' by Infinity Fabric(tm) if they are targeted at non-write-cachable memory, e.g., [fine-grained memory](Mtype) allocations or [uncached memory](Mtype) allocations on the [MI2XX](2xxnote).
     - Percent
   * - Atomic Traffic
     - The percent of write requests generated by the L2 cache that are atomic requests to _any_ memory location.  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric Write and Atomic bandwidth that is due to use of atomics. Note that on current CDNA accelerators, such as the [MI2XX](2xxnote), requests are only considered 'atomic' by Infinity Fabric(tm) if they are targeted at [fine-grained memory](Mtype) allocations or [uncached memory](Mtype) allocations.
     - Percent
   * - Uncached Write and Atomic Traffic
     - The percent of write and atomic requests generated by the L2 cache that are targeting [uncached memory allocations](Mtype).  This breakdown does not consider the _size_ of the request (i.e., 32B and 64B requests are both counted as a single request), so this metric only _approximates_ the percent of the L2-Fabric read bandwidth directed to uncached memory allocations.
     - Percent
   * - Read Latency
     - The time-averaged number of cycles read requests spent in Infinity Fabric(tm) before data was returned to the L2.
     - Cycles
   * - Write Latency
     - The time-averaged number of cycles write requests spent in Infinity Fabric(tm) before a completion acknowledgement was returned to the L2.
     - Cycles
   * - Atomic Latency
     - The time-averaged number of cycles atomic requests spent in Infinity Fabric(tm) before a completion acknowledgement (atomic without return value) or data (atomic with return value) was returned to the L2.
     - Cycles
   * - Read Stall
     - The ratio of the total number of cycles the L2-Fabric interface was stalled on a read request to any destination (local HBM, remote PCIe(r) connected accelerator / CPU, or remote Infinity Fabric(tm) connected accelerator{sup}`1` / CPU) over the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Write Stall
     - The ratio of the total number of cycles the L2-Fabric interface was stalled on a write or atomic request to any destination (local HBM, remote accelerator / CPU, PCIe(r) connected accelerator / CPU, or remote Infinity Fabric(tm) connected accelerator{sup}`1` / CPU) over the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent

(L2_req_metric_details)= #### Detailed Transaction Metrics

The following metrics are available in the detailed L2-Fabric
transaction breakdown table:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - 32B Read Requests
     - The total number of L2 requests to Infinity Fabric(tm) to read 32B of data from any memory location, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail. Typically unused on CDNA accelerators.
     - Requests per [normalization-unit](normunit)
   * - Uncached Read Requests
     - The total number of L2 requests to Infinity Fabric(tm) to read [uncached data](Mtype) from any memory location, per [normalization-unit](normunit). 64B requests for uncached data are counted as two 32B uncached data requests. See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - 64B Read Requests
     - The total number of L2 requests to Infinity Fabric(tm) to read 64B of data from any memory location, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - HBM Read Requests
     - The total number of L2 requests to Infinity Fabric(tm) to read 32B or 64B of data from the accelerator's local HBM, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Remote Read Requests
     - The total number of L2 requests to Infinity Fabric(tm) to read 32B or 64B of data from any source other than the accelerator's local HBM, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - 32B Write and Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to write or atomically update 32B of data to any memory location, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Uncached Write and Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to write or atomically update 32B or 64B of [uncached data](Mtype), per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - 64B Write and Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to write or atomically update 64B of data in any memory location, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - HBM Write and Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to write or atomically update 32B or 64B of data in the accelerator's local HBM, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Remote Write and Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to write or atomically update 32B or 64B of data in any memory location other than the accelerator's local HBM, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Atomic Requests
     - The total number of L2 requests to Infinity Fabric(tm) to atomically update 32B or 64B of data in any memory location, per [normalization-unit](normunit). See [request-flow](L2_req_flow) for more detail.  Note that on current CDNA accelerators, such as the [MI2XX](2xxnote), requests are only considered 'atomic' by Infinity Fabric(tm) if they are targeted at non-write-cachable memory, e.g., [fine-grained memory](Mtype) allocations or [uncached memory](Mtype) allocations on the [MI2XX](2xxnote).
     - Requests per [normalization-unit](normunit)

L2-Fabric Interface Stalls
~~~~~~~~~~~~~~~~~~~~~~~~~~

When the interface between the L2 cache and Infinity Fabric(tm) becomes
backed up by requests, it may stall preventing the L2 from issuing
additional requests to Infinity Fabric(tm) until prior requests
complete. This section gives a breakdown of what types of requests in a
kernel caused a stall (e.g., read vs write), and to which locations
(e.g., to the accelerator’s local memory, or to remote
accelerators/CPUs).

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Read - PCIe(r) Stall
     - The number of cycles the L2-Fabric interface was stalled on read requests to remote PCIe(r) connected accelerators{sup}`1` or CPUs as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Read - Infinity Fabric(tm) Stall
     - The number of cycles the L2-Fabric interface was stalled on read requests to remote Infinity Fabric(tm) connected accelerators{sup}`1` or CPUs as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Read - HBM Stall
     - The number of cycles the L2-Fabric interface was stalled on read requests to the accelerator's local HBM as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Write - PCIe(r) Stall
     - The number of cycles the L2-Fabric interface was stalled on write or atomic requests to remote PCIe(r) connected accelerators{sup}`1` or CPUs as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Write - Infinity Fabric(tm) Stall
     - The number of cycles the L2-Fabric interface was stalled on write or atomic requests to remote Infinity Fabric(tm) connected accelerators{sup}`1` or CPUs as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Write - HBM Stall
     - The number of cycles the L2-Fabric interface was stalled on write or atomic requests to accelerator's local HBM as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent
   * - Write - Credit Starvation
     - The number of cycles the L2-Fabric interface was stalled on write or atomic requests to any memory location because too many write/atomic requests were currently in flight, as a percent of the [total active L2 cycles](TotalActiveL2Cycles).
     - Percent

.. code:: {note}

   {sup}`1` In addition to being used for on-accelerator data-traffic, AMD [Infinity Fabric](https://www.amd.com/en/technologies/infinity-architecture)(tm) technology can be used to connect multiple accelerators to achieve advanced peer-to-peer connectivity and enhanced bandwidths over traditional PCIe(r) connections.
   Some AMD Instinct(tm) MI accelerators, e.g., the MI250X, [feature coherent CPU↔accelerator connections built using AMD Infinity Fabric(tm)](https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf)

.. code:: {warning}

   On current CDNA accelerators and GCN GPUs, these L2↔Fabric stalls can be undercounted in some circumstances.

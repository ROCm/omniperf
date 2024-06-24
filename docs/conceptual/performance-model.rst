.. meta::
   :description: Omniperf documentation and reference
   :keywords: Omniperf, ROCm, performance, model, profiler, tool, Instinct,
   accelerator, AMD

*****************
Performance model
*****************

Omniperf makes available an extensive list of metrics to better understand
achieved application performance on AMD Instinct™ MI-series accelerators
including Graphics Core Next (GCN) GPUs like the AMD Instinct MI50, CDNA
accelerators like the MI100, and CDNA2 accelerators such as the MI250X, MI250,
and MI210.

To best use profiling data, it's important to understand the role of various
hardware blocks of AMD Instinct accelerators. This section describes each
hardware block on the accelerator as interacted with by a software developer to
give a deeper understanding of the metrics reported by profiling data. Refer to
:doc:`<how-to/profile-mode>` for more practical examples and details on how to use
Omniperf to optimize your code.

.. note::

   In this guide, **MI2XX** refers to any of the CDNA2 architecture-based AMD
   Instinct MI250X, MI250, and MI210 accelerators interchangeably in cases
   where the exact product at hand is not vital.

   For a comparison of AMD Instinct accelerator specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`. For product
   details, see the
   `MI250X <https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x>`_,
   `MI250 <https://www.amd.com/en/products/accelerators/instinct/mi200/mi250>`_,
   and
   `MI210 <https://www.amd.com/en/products/accelerators/instinct/mi200/mi210>`_
   product pages.

.. include:: ./includes/compute-unit.rst

Pipeline descriptions
=====================

.. _perf-model-valu:

Vector arithmetic logic unit (VALU)
-----------------------------------

The vector arithmetic logic unit (VALU) executes vector instructions over an
entire wavefront, with each work-item or vector-lane potentially operating on
distinct data.

The VALU of a CDNA accelerator or GCN GPU typically consists of:

* Four 16-wide SIMD processors (see [An introduction to AMD GPU
Programming with HIP](https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf) for more details)
* Four 64 or 128 KiB VGPR files (yielding a total of 256-512 KiB total per CU), see [AGPRs](agprs) for more detail.
* An instruction buffer (per-SIMD) that contains execution slots for up to 8 wavefronts (for 32 total wavefront slots on each CU).
* A vector memory (VMEM) unit which transfers data between VGPRs and memory; each work-item supplies its own memory address and supplies or receives unique data.
* CDNA accelerators, such as the MI100 and [MI2XX](2xxnote), contain additional [Matrix Fused Multiply-Add (MFMA) units](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/).

In order to support branching / conditionals, each wavefront in the VALU has a
distinct execution mask which determines which work-items in the wavefront are
active for the currently executing instruction.
When executing a VALU instruction, inactive work-items (according to the current
execution mask of the wavefront) do not execute the instruction and are treated
as no-ops.

.. _perf-model-salu:

Scalar arithmetic logic unit (SALU)
-----------------------------------

The scalar arithmetic logic unit (SALU) executes instructions shared between all
work-items in a wavefront. This includes control-flow -- such as
if/else conditionals, branches and looping -- pointer arithmetic, loading common
values, etc.

The SALU consists of:

- A scalar processor capable of various arithmetic, conditional, and comparison
  (etc.) operations. See, e.g., [Chapter 5. Scalar ALU Operations](https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf) of the CDNA2 Instruction Set Architecture (ISA) Guide for more detail.
- A 12.5 KiB Scalar General Purpose Register (SGPR) file
- A scalar memory (SMEM) unit which transfers data between SGPRs and memory

Data loaded by the SMEM can be cached in the [scalar L1 data cache](sL1D), and is typically only used for read-only, uniform accesses such as kernel arguments, or HIP's `__constant__` memory.

.. _perf-model-lds:

Local data share (LDS)
----------------------

.. _perf-model-branch:

The local data share (LDS, a.k.a., "shared memory") is fast on-CU scratchpad that can be explicitly managed by software to effectively share data and to coordinate between wavefronts in a workgroup.

```{figure} images/lds.*
:scale: 150 %
:alt: Performance model of the Local Data Share (LDS) on AMD Instinct(tm) MI accelerators.
:align: center

Performance model of the Local Data Share (LDS) on AMD Instinct(tm) MI accelerators.
```

Above is Omniperf's performance model of the LDS on CDNA accelerators (adapted from [GCN Architecture, by Mike Mantor](https://old.hotchips.org/wp-content/uploads/hc_archives/hc24/HC24-3-ManyCore/HC24.28.315-AMD.GCN.mantor_v1.pdf), slide 20).
The SIMDs in the [VALU](valu) are connected to the LDS in pairs (see above).
Only one SIMD per pair may issue an LDS instruction at a time, but both pairs may issue concurrently.

On CDNA accelerators, the LDS contains 32 banks and each bank is 4B wide.
The LDS is designed such that each bank can be read from/written to/atomically updated every cycle, for a total throughput of 128B/clock ([GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah), slide 40).

On each of the two ports to the SIMDs, 64B can be sent in each direction per cycle. So, a single wavefront, coming from one of the 2 SIMDs in a pair, can only get back 64B/cycle (16 lanes per cycle). The input port is shared between data and address and this can affect achieved bandwidth for different data sizes. For example, a 64-wide store where each lane is sending a 4B value takes 8 cycles (50% peak bandwidth) while a 64-wide store where each lane is sending a 16B value takes 20 cycles (80% peak bandwidth).

In addition, the LDS contains conflict-resolution hardware to detect and handle bank conflicts.
A bank conflict occurs when two (or more) work-items in a wavefront want to read, write, or atomically update different addresses that map to the same bank in the same cycle.
In this case, the conflict detection hardware will determine a new schedule such that the access is split into multiple cycles with no conflicts in any single cycle.

When multiple work-items want to read from the same address within a bank, the result can be efficiently broadcasted ([GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah), slide 41).
Multiple work-items writing to the same address within a bank typically results undefined behavior in HIP and other languages, as the LDS will write the value from the last work-item as determined by the hardware scheduler ([GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah), slide 41).  This behavior may be useful in the very specific case of storing a uniform value.

Relatedly, an address conflict is defined as occurring when two (or more) work-items in a wavefront want to atomically update the same address on the same cycle.
As in a bank-conflict, this may cause additional cycles of work for the LDS operation to complete.

Branch
------

The branch unit is responsible for executing jumps and branches to execute
control flow operations.
Note that Branch operations are not used for execution mask updates, but only
for “whole wavefront” control-flow changes.

.. _perf-model-scheduler:

Scheduler
---------

The scheduler is responsible for arbitration and issue of instructions for all the wavefronts currently executing on the CU.  On every clock cycle, the scheduler:

- considers waves from one of the SIMD units for execution, selected in a round-robin fashion between the SIMDs in the [compute unit](CU)
- issues up to one instruction per wavefront on the selected SIMD
- issues up to one instruction per each of the instruction categories among the waves on the selected SIMD:
  - [VALU](valu)
  - [VMEM](valu) operations
  - [SALU](salu) / SMEM operations
  - [LDS](lds)
  - [Branch](branch) operations

This gives a maximum of five issued Instructions Per Cycle (IPC), per-SIMD, per-CU ([AMD GPU HIP Training](https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf), [GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah)).

On CDNA accelerators with [MFMA](mfma) instructions, these are issued via the [VALU](valu). Some of them will execute on a separate functional unit and typically allow other [VALU](valu) operations to execute in their shadow (see the [MFMA](mfma) section for more detail).

```{note}
The IPC model used by Omniperf omits the following two complications for clarity.
First, CDNA accelerators contain other execution units on the CU that are unused for compute applications.
Second, so-called "internal" instructions (see [Layla Mah's GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah), slide 29) are not issued to a functional unit, and can technically cause the maximum IPC to _exceed_ 5 instructions per-cycle in special (largely unrealistic) cases.
The latter issue is discussed in more detail in our ['internal' IPC](Internal_ipc) example.
```

.. _perf-model-mfma:

Matrix fused multiply-add (MFMA)
--------------------------------

Pipeline metrics
================

.. _perf-model-wavefront:

Wavefront
---------

Wavefront runtime stats
^^^^^^^^^^^^^^^^^^^^^^^

The wavefront runtime statistics give a high-level overview of the execution of
wavefronts in a kernel:

Instruction mix
---------------

Overall instruction mix
^^^^^^^^^^^^^^^^^^^^^^^

VALU instruction mix
^^^^^^^^^^^^^^^^^^^^

VMEM instruction mix
^^^^^^^^^^^^^^^^^^^^

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
:doc:`<how-to/profiling>` for more practical
examples and details on how to use Omniperf to optimize your code.

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

.. _def-compute-unit:

Compute unit
============

The compute unit (CU) is responsible for executing a user's kernels on
CDNA-based accelerators. All :ref:`wavefronts` of a :ref:`workgroup` are
scheduled on the same CU.

.. image:: ../data/performance-model/gcn_compute_unit.png
    :alt: AMD CDNA accelerator compute unit diagram

The CU consists of several independent pipelines and functional units.

* The *vector arithmetic logic unit (VALU)* is composed of multiple SIMD (single
  instruction, multiple data) vector processors, vector general purpose
  registers (VGPRs) and instruction buffers. The VALU is responsible for
  executing much of the computational work on CDNA accelerators, including but
  not limited to floating-point operations (FLOPs) and integer operations
  (IOPs).
* The *vector memory (VMEM)* unit is responsible for issuing loads, stores and
  atomic operations that interact with the memory system.
* The *scalar arithmetic logic unit (SALU)* is shared by all threads in a
  [wavefront](wavefront), and is responsible for executing instructions that are
  known to be uniform across the wavefront at compile-time. The SALU has a
  memory unit (SMEM) for interacting with memory, but it cannot issue separately
  from the SALU.
* The *local data share (LDS)* is an on-CU software-managed scratchpad memory
  that can be used to efficiently share data between all threads in a
  [workgroup](workgroup).
* The *scheduler* is responsible for issuing and decoding instructions for all
  the [wavefronts](wavefront) on the compute unit.
* The *vector L1 data cache (vL1D)* is the first level cache local to the
  compute unit. On current CDNA accelerators, the vL1D is write-through. The
  vL1D caches from multiple compute units are kept coherent with one another
  through software instructions.
* CDNA accelerators -- that is, AMD Instinct MI100 and newer -- contain
  specialized matrix-multiplication accelerator pipelines known as the
  [matrix-fused multiply-add (MFMA)](mfma).

For a more in-depth description of a compute unit on a CDNA accelerator, see
slides 22 to 28 in
`An introduction to AMD GPU Programming with HIP <https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf>`_
and slide 27 in
`The AMD GCN Architecture - A Crash Course (Layla Mah) <https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah>`_.

Pipeline descriptions
=====================

.. _perf-model-valu:

Vector arithmetic logic unit (VALU)
-----------------------------------

.. _perf-model-salu:

Scalar arithmetic logic unit (SALU)
-----------------------------------

.. _perf-model-lds:

Local data share (LDS)
----------------------

.. _perf-model-branch:

Branch
------

.. _perf-model-scheduler:

Scheduler
---------

.. _perf-model-mfma:

Matrix-fused multiply add (MFMA)
--------------------------------

Pipeline metrics
================

.. _perf-model-wavefront:

Wavefront
---------

Wavefront runtime stats
^^^^^^^^^^^^^^^^^^^^^^^

Instruction mix
---------------

Overall instruction mix
^^^^^^^^^^^^^^^^^^^^^^^

VALU instruction mix
^^^^^^^^^^^^^^^^^^^^

VMEM instruction mix
^^^^^^^^^^^^^^^^^^^^

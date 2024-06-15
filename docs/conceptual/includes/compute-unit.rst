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
  [matrix fused multiply-add (MFMA)](mfma).

For a more in-depth description of a compute unit on a CDNA accelerator, see
slides 22 to 28 in
`An introduction to AMD GPU Programming with HIP <https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf>`_
and slide 27 in
`The AMD GCN Architecture - A Crash Course (Layla Mah) <https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah>`_.

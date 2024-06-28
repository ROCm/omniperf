*****************
Compute unit (CU)
*****************

The compute unit (CU) is responsible for executing a user's kernels on
CDNA-based accelerators. All :ref:`wavefronts` of a :ref:`workgroup` are
scheduled on the same CU.

.. image:: ../data/performance-model/gcn_compute_unit.png
    :alt: AMD CDNA accelerator compute unit diagram

The CU consists of several independent execution pipelines and functional units.

* The :ref:`desc-valu` is composed of multiple SIMD (single
  instruction, multiple data) vector processors, vector general purpose
  registers (VGPRs) and instruction buffers. The VALU is responsible for
  executing much of the computational work on CDNA accelerators, including but
  not limited to floating-point operations (FLOPs) and integer operations
  (IOPs).
* The *vector memory (VMEM)* unit is responsible for issuing loads, stores and
  atomic operations that interact with the memory system.
* The :ref:`desc-salu` is shared by all threads in a
  [wavefront](wavefront), and is responsible for executing instructions that are
  known to be uniform across the wavefront at compile-time. The SALU has a
  memory unit (SMEM) for interacting with memory, but it cannot issue separately
  from the SALU.
* The :ref:`desc-lds` is an on-CU software-managed scratchpad memory
  that can be used to efficiently share data between all threads in a
  [workgroup](workgroup).
* The :ref:`desc-scheduler` is responsible for issuing and decoding instructions for all
  the [wavefronts](wavefront) on the compute unit.
* The *vector L1 data cache (vL1D)* is the first level cache local to the
  compute unit. On current CDNA accelerators, the vL1D is write-through. The
  vL1D caches from multiple compute units are kept coherent with one another
  through software instructions.
* CDNA accelerators -- that is, AMD Instinct MI100 and newer -- contain
  specialized matrix-multiplication accelerator pipelines known as the
  :ref:`desc-mfma`.

For a more in-depth description of a compute unit on a CDNA accelerator, see
:hip-training-2019:`22` and :gcn-crash-course:`27`.

:ref:`pipeline-desc` details the various
execution pipelines (VALU, SALU, LDS, Scheduler, etc.). The metrics
presented by Omniperf for these pipelines are described in
:ref:`pipeline-metrics`. Finally, the `vL1D <vL1D>`__ cache and
:ref:`LDS <desc-lds>` will be described their own sections.

.. include:: ./includes/pipeline-descriptions.rst

.. include:: ./includes/pipeline-metrics.rst

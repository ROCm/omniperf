.. meta::
   :description: Omniperf terminology and definitions
   :keywords: Omniperf, ROCm, glossary, definitions, terms, profiler, tool,
              Instinct, accelerator, AMD

***********
Definitions
***********

The following table briefly defines some terminology used in Omniperf interfaces
and in this documentation.

.. include:: ./includes/terms.rst

.. include:: ./includes/normalization-units.rst

.. _memory-spaces:

Memory spaces
=============

AMD Instinct MI accelerators can access memory through multiple address spaces
which may map to different physical memory locations on the system. The
following table provides a view into how various types of memory used
in HIP map onto these constructs:

.. list-table::
   :header-rows: 1

   * - LLVM Address Space
     - Hardware Memory Space
     - HIP Terminology

   * - Generic
     - Flat
     - N/A

   * - Global
     - Global
     - Global

   * - Local
     - LDS
     - LDS/Shared

   * - Private
     - Scratch
     - Private

   * - Constant
     - Same as global
     - Constant

The following is a high-level description of the address spaces in the AMDGPU
backend of LLVM:

.. list-table::
   :header-rows: 1

   * - Address space
     - Description

   * - Global
     - Memory that can be seen by all threads in a process, and may be backed by
       the local accelerator's HBM, a remote accelerator's HBM, or the CPU's
       DRAM.

   * - Local
     - Memory that is only visible to a particular workgroup. On AMD's Instinct
       accelerator hardware, this is stored in :doc:`LDS <local-data-share>`
       memory.

   * - Private
     - Memory that is only visible to a particular [work-item](workitem)
       (thread), stored in the scratch space on AMD's Instinct accelerators.

   * - Constant
     - Read-only memory that is in the global address space and stored on the
       local accelerator's HBM.

   * - Generic
     - Used when the compiler cannot statically prove that a pointer is
       addressing memory in a single (non-generic) address space. Mapped to Flat
       on AMD's Instinct accelerators, the pointer could dynamically address
       global, local, private or constant memory.

`LLVM's documentation for AMDGPU Backend <https://llvm.org/docs/AMDGPUUsage.html#address-spaces>`_
has the most up-to-date information. Refer to this source for a more complete
explanation.

.. _memory-type:

Memory type
===========

AMD Instinct accelerators contain a number of different memory allocation
types to enable the HIP language's
:doc:`memory coherency model <hip:how-to/programming_manual>`.
These memory types are broadly similar between AMD Instinct accelerator
generations, but may differ in exact implementation.

In addition, these memory types *might* differ between accelerators on the same
system, even when accessing the same memory allocation.

For example, an :ref:`MI2XX <mixxx-note>` accelerator accessing *fine-grained*
memory allocated local to that device may see the allocation as coherently
cacheable, while a remote accelerator might see the same allocation as
*uncached*.


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
:doc:`../how-to/profile/mode` for more practical examples and details on how
to use Omniperf to optimize your code.

.. _mixxx-note:

.. note::

   In this chapter, **MI2XX** refers to any of the CDNA2 architecture-based AMD
   Instinct MI250X, MI250, and MI210 accelerators interchangeably in cases
   where the exact product at hand is not vital.

   For a comparison of AMD Instinct accelerator specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`. For product
   details, see the :prod-page:`MI250X <mi200/mi250x>`,
   :prod-page:`MI250 <mi200/mi250>`, and :prod-page:`MI210 <mi200/mi210>`
   product pages.

In this chapter, the way Omniperf models performance is explained in detail
throughout the following sections.

* :doc:`compute-unit`

* :doc:`l2-cache`

* :doc:`shader-engine`

* :doc:`command-processor`

* :doc:`system-speed-of-light`

.. _perf-model-ext-refs:

References
==========

Some sections in this chapter cite the following publicly available
documentation.

* :hip-training-pdf:`Introduction to AMD GPU Programming with HIP <>`

* :mi200-isa-pdf:`CDNA2 ISA Reference Guide <>`

* :cdna2-white-paper:`CDNA2 white paper <>`

* :hsa-runtime-pdf:`HSA Runtime Programmer's Reference Manual <>`

* :gcn-crash-course:`The AMD GCN Architecture - A Crash Course (Layla Mah) <>`

* :mantor-gcn-pdf:`AMD Radeon HD7970 with GCN Architecture <>`

* :mantor-vega10-pdf:`AMD Radeon Next Generation GPU Architecture - Vega10 <>`

* :llvm-docs:`LLVM User Guide for AMDGPU Backend <>`

.. rubric:: Disclaimer

PCIe® is a registered trademark of PCI-SIG Corporation.

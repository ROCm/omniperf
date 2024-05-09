*****************
What is Omniperf?
*****************

Omniperf is a kernel-level profiling tool for machine learning and HPC workloads running on AMD Instinct™ accelerators.

AMD Instinct MI-series accelerators are data center-class GPUs designed for compute and have some graphics capabilities
disabled or removed. The Omniperf tool targets use with
:doc:`accelerators in the MI100, MI200, and MI300 families <rocm:conceptual/gpu-arch>`, primarily. Development is in
progress to support Radeon™ (RDNA) GPUs.

Omniperf is currently built on top of :doc:`rocprof <rocprofiler:rocprofv1>` to monitor hardware performance counters.

Features
========

Omniperf performs profiling based on all available hardware counters for the target accelerator. It provides high-level
performance analysis features including System Speed of Light (SOL), Hardware block-level SOL, Memory Chart
Analysis, Roofline Analysis, Baseline Comparisons, and more.

Omniperf supports both command line analysis and GUI analysis.

Detailed feature list:

- MI100 support
- MI200 support
- MI300 support
- Standalone GUI Analyzer
- Grafana/MongoDB GUI Analyzer
- Dispatch Filtering
- Kernel Filtering
- GPU ID Filtering
- Baseline Comparison
- Multi-Normalizations
- System Info Panel
- System Speed-of-Light Panel
- Kernel Statistic Panel
- Memory Chart Analysis Panel
- Roofline Analysis Panel (*Supported on MI200 only, Ubuntu 20.04, SLES 15 SP3 or RHEL8*)
- Command Processor (CP) Panel
- Workgroup Manager (SPI) Panel
- Wavefront Launch Panel
- Compute Unit - Instruction Mix Panel
- Compute Unit - Pipeline Panel
- Local Data Share (LDS) Panel
- Instruction Cache Panel
- Scalar L1D Cache Panel
- L1 Address Processing Unit, a.k.a. Texture Addresser (TA) / L1 Backend Data Processing Unit, a.k.a. Texture Data (TD)
  panel(s)
- Vector L1D Cache Panel
- L2 Cache Panel
- L2 Cache (per-Channel) Panel

Compatible SoCs
===============

.. list-table::
    :header-rows: 1

    * - Platform
      - Status

    * - Vega 20 (MI50/60)
      - No support

    * - MI100
      - Supported

    * - MI200
      - Supported

    * - MI300
      - Supported

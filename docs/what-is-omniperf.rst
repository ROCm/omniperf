.. meta::
   :description: What is Omniperf?
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD

*****************
What is Omniperf?
*****************

Omniperf is a kernel-level profiling tool for machine learning and HPC workloads
running on AMD Instinct™ accelerators.

AMD Instinct MI-series accelerators are data center-class GPUs designed for
compute and have some graphics capabilities disabled or removed. Omniperf
primarily targets use with
:doc:`accelerators in the MI300, MI200, and MI100 families <rocm:conceptual/gpu-arch>`.
Development is in progress to support Radeon™ (RDNA) GPUs.

Omniperf is built on top of :doc:`ROCProfiler <rocprofiler:rocprofv1>` to
monitor hardware performance counters.

Omniperf features
=================

Omniperf offers comprehensive profiling based on all available hardware counters
for the target accelerator. It delivers advanced performance analysis features,
such as system speed-of-light (SOL) and hardware block-level SOL evaluations.
Additionally, Omniperf provides in-depth memory chart analysis, roofline
analysis, baseline comparisons, and more, ensuring a thorough understanding of
system performance.

Omniperf supports both command line analysis and GUI analysis.

Detailed feature list
---------------------

* :doc:`Support for MI300, MI200, and MI100 <reference/compatible-accelerators>`
* Standalone GUI analyzer
* GUI analyzer via Grafana and MongoDB
* Dispatch filtering
* Kernel filtering
* GPU ID filtering
* Baseline comparison
* Multiple normalizations
* *System info* panel
* *System Speed-of-Light* panel
* *Kernel Statistic* panel
* *Memory Chart Analysis* panel
* *Roofline Analysis* panel (*Supported on MI200 only, Ubuntu 20.04, SLES 15 SP3 or RHEL8*)
* *Command Processor (CP)* panel
* *Workgroup Manager (SPI)* panel
* *Wavefront Launch* Panel
* *Compute Unit - Instruction Mix* panel
* *Compute Unit - Pipeline* panel
* *Local Data Share (LDS)* panel
* *Instruction Cache* panel
* *Scalar L1D Cache* panel
* *L1 Address Processing Unit*, or, *Texture Addresser (TA)* and
 *L1 Backend Data Processing Unit*, or, *Texture Data (TD)* panels
* *Vector L1D Cache* panel
* *L2 Cache* panel
* *L2 Cache (per-channel)* panel


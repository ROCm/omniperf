(SE)= ## Shader Engine (SE)

The `CUs <CU>`__ on a CDNA accelerator are grouped together into a
higher-level organizational unit called a Shader Engine (SE):

\```{figure} images/selayout.png :alt: Example of CU-grouping into
shader-engines on AMD Instinct(tm) MI accelerators. :align: center
:name: selayout-fig

Example of CU-grouping into shader-engines on AMD Instinct(tm) MI
accelerators.

::


   The number of CUs on a SE varies from chip-to-chip (see, for example [AMD GPU HIP Training](https://www.olcf.ornl.gov/wp-content/uploads/2019/09/AMD_GPU_HIP_training_20190906.pdf), slide 20).
   In addition, newer accelerators such as the AMD Instinct(tm) MI 250X have 8 SEs per accelerator.

   For the purposes of Omniperf, we consider resources that are shared between multiple CUs on a single SE as part of the SE's metrics.
   These include:
     - the [scalar L1 data cache](sL1D)
     - the [L1 instruction cache](L1I)
     - the [workgroup manager](SPI)

   (sL1D)=
   ### Scalar L1 Data Cache (sL1D)

   The Scalar L1 Data cache (sL1D) can cache data accessed from scalar load instructions (and scalar store instructions on architectures where they exist) from wavefronts in the [CUs](CU).
   The sL1D is shared between multiple CUs ([GCN Crash Course](https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah), slide 36) --- the exact number of CUs depends on the architecture in question (3 CUs in GCN GPUs and MI100, 2 CUs in [MI2XX](2xxnote)) --- and is backed by the [L2](L2) cache.

   In typical usage, the data in the sL1D is comprised of (e.g.,):
     - Kernel arguments, e.g., pointers, [non-populated](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-sgpr-register-set-up-order-table) grid/block dimensions, etc.
     - HIP's `__constant__` memory, when accessed in a provably uniform{sup}`1` manner
     - Other memory, when accessed in a provably uniform manner, *and* the backing memory is provably constant{sup}`1`

   ```{note}
   {sup}`1`
   The scalar data cache is used when the compiler emits scalar loads to access data.
   This requires that the data be _provably_ uniformly accessed (i.e., the compiler can verify that all work-items in a wavefront access the same data), _and_ that the data can be proven to be read-only (e.g., HIP's `__constant__` memory, or properly `__restrict__`'ed pointers to avoid write-aliasing).
   Access of e.g., `__constant__` memory is not guaranteed to go through the sL1D if, e.g., the wavefront loads a non-uniform value.

(sL1D_SOL)= #### Scalar L1D Speed-of-Light

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

The Scalar L1D speed-of-light chart shows some key metrics of the sL1D
cache as a comparison with the peak achievable values of those metrics:

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Bandwidth
     - The number of bytes looked up in the sL1D cache, as a percent of the peak theoretical bandwidth. Calculated as the ratio of sL1D requests over the [total sL1D cycles](TotalSL1DCycles). 
     - Percent
   * - Cache Hit Rate
     - The percent of sL1D requests that hit{sup}`1` on a previously loaded line in the cache. Calculated as the ratio of the number of sL1D requests that hit over the number of all sL1D requests.
     - Percent
   * - sL1D-L2 BW
     - The number of bytes requested by the sL1D from the L2 cache, as a percent of the peak theoretical sL1D → L2 cache bandwidth.  Calculated as the ratio of the total number of requests from the sL1D to the L2 cache over the [total sL1D-L2 interface cycles](TotalSL1DCycles).
     - Percent

.. code:: {note}

   {sup}`1` Unlike the [vL1D](vL1D) and [L2](L2) caches, the sL1D cache on AMD Instinct(tm) MI CDNA accelerators does _not_ use "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'duplicated miss' (see below).

Scalar L1D Cache Accesses
^^^^^^^^^^^^^^^^^^^^^^^^^

This panel gives more detail on the types of accesses made to the sL1D,
and the hit/miss statistics.

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Requests
     - The total number of requests, of any size or type, made to the sL1D per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Hits
     - The total number of sL1D requests that hit on a previously loaded cache line, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Misses - Non Duplicated
     - The total number of sL1D requests that missed on a cache line that *was not* already pending due to another request, per [normalization-unit](normunit). See note in [speed-of-light section](sL1D_SOL) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Misses - Duplicated
     - The total number of sL1D requests that missed on a cache line that *was* already pending due to another request, per [normalization-unit](normunit). See note in [speed-of-light section](sL1D_SOL) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Cache Hit Rate
     - Indicates the percent of sL1D requests that hit on a previously loaded line the cache. The ratio of the number of sL1D requests that hit{sup}`1` over the number of all sL1D requests.
     - Percent
   * - Read Requests (Total)
     - The total number of sL1D read requests of any size, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Atomic Requests
     - The total number of sL1D atomic requests of any size, per [normalization-unit](normunit).  Typically unused on CDNA accelerators.
     - Requests per [normalization-unit](normunit)
   * - Read Requests (1 DWord)
     - The total number of sL1D read requests made for a single dword of data (4B), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Read Requests (2 DWord)
     - The total number of sL1D read requests made for a two dwords of data (8B), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Read Requests (4 DWord)
     - The total number of sL1D read requests made for a four dwords of data (16B), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Read Requests (8 DWord)
     - The total number of sL1D read requests made for a eight dwords of data (32B), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Read Requests (16 DWord)
     - The total number of sL1D read requests made for a sixteen dwords of data (64B), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)

.. code:: {note}

   {sup}`1`Unlike the [vL1D](vL1D) and [L2](L2) caches, the sL1D cache on AMD Instinct(tm) MI CDNA accelerators does _not_ use "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'duplicated miss' (see below).

sL1D ↔ L2 Interface
^^^^^^^^^^^^^^^^^^^

This panel gives more detail on the data requested across the
sL1D↔\ `L2 <L2>`__ interface.

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - sL1D-L2 BW
     - The total number of bytes read from/written to/atomically updated across the sL1D↔[L2](L2) interface, per [normalization-unit](normunit).  Note that sL1D writes and atomics are typically unused on current CDNA accelerators, so in the majority of cases this can be interpreted as an sL1D→L2 read bandwidth.
     - Bytes per [normalization-unit](normunit)
   * - Read Requests
     - The total number of read requests from sL1D to the [L2](L2), per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Write Requests
     - The total number of write requests from sL1D to the [L2](L2), per [normalization-unit](normunit).  Typically unused on current CDNA accelerators.
     - Requests per [normalization-unit](normunit)
   * - Atomic Requests
     - The total number of atomic requests from sL1D to the [L2](L2), per [normalization-unit](normunit).  Typically unused on current CDNA accelerators.
     - Requests per [normalization-unit](normunit)
   * - Stall Cycles
     - The total number of cycles the sL1D↔[L2](L2) interface was stalled, per [normalization-unit](normunit).
     - Cycles per [normalization-unit](normunit)

(L1I)= ### L1 Instruction Cache (L1I)

As with the `sL1D <sL1D>`__, the L1 Instruction (L1I) cache is shared
between multiple CUs on a shader-engine, where the precise number of CUs
sharing a L1I depends on the architecture in question (`GCN Crash
Course <https://www.slideshare.net/DevCentralAMD/gs4106-the-amd-gcn-architecture-a-crash-course-by-layla-mah>`__,
slide 36) and is backed by the `L2 <L2>`__ cache. Unlike the sL1D, the
instruction cache is read-only.

(L1I_SOL)= #### L1I Speed-of-Light

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

The L1 Instruction Cache speed-of-light chart shows some key metrics of
the L1I cache as a comparison with the peak achievable values of those
metrics:

.. code:: {list-table}

   :header-rows: 1
   :widths: 15 70 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Bandwidth
     - The number of bytes looked up in the L1I cache, as a percent of the peak theoretical bandwidth. Calculated as the ratio of L1I requests over the [total L1I cycles](TotalL1ICycles). 
     - Percent
   * - Cache Hit Rate
     - The percent of L1I requests that hit on a previously loaded line the cache. Calculated as the ratio of the number of L1I requests that hit{sup}`1` over the number of all L1I requests.
     - Percent
   * - L1I-L2 BW
     - The percent of the peak theoretical L1I → L2 cache request bandwidth achieved.  Calculated as the ratio of the total number of requests from the L1I to the L2 cache over the [total L1I-L2 interface cycles](TotalL1ICycles).
     - Percent
   * - Instruction Fetch Latency
     - The average number of cycles spent to fetch instructions to a [CU](cu).
     - Cycles

.. code:: {note}

   {sup}`1`Unlike the [vL1D](vL1D) and [L2](L2) caches, the L1I cache on AMD Instinct(tm) MI CDNA accelerators does _not_ use "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'duplicated miss' (see below).

L1I Cache Accesses
^^^^^^^^^^^^^^^^^^

This panel gives more detail on the hit/miss statistics of the L1I:

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Requests
     - The total number of requests made to the L1I per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Hits
     - The total number of L1I requests that hit on a previously loaded cache line, per [normalization-unit](normunit).
     - Requests per [normalization-unit](normunit)
   * - Misses - Non Duplicated
     - The total number of L1I requests that missed on a cache line that *was not* already pending due to another request, per [normalization-unit](normunit). See note in [speed-of-light section](L1I_SOL) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Misses - Duplicated
     - The total number of L1I requests that missed on a cache line that *was* already pending due to another request, per [normalization-unit](normunit). See note in [speed-of-light section](L1I_SOL) for more detail.
     - Requests per [normalization-unit](normunit)
   * - Cache Hit Rate
     - The percent of L1I requests that hit{sup}`1` on a previously loaded line the cache. Calculated as the ratio of the number of L1I requests that hit over the the number of all L1I requests.
     - Percent

.. code:: {note}

   {sup}`1`Unlike the [vL1D](vL1D) and [L2](L2) caches, the L1I cache on AMD Instinct(tm) MI CDNA accelerators does _not_ use "hit-on-miss" approach to reporting cache hits.
   That is, if while satisfying a miss, another request comes in that would hit on the same pending cache line, the subsequent request will be counted as a 'duplicated miss' (see below).

L1I - L2 Interface
^^^^^^^^^^^^^^^^^^

This panel gives more detail on the data requested across the
L1I-`L2 <L2>`__ interface.

.. code:: {list-table}

   :header-rows: 1
   :widths: 18 65 17
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - L1I-L2 BW
     - The total number of bytes read across the L1I-[L2](L2) interface, per [normalization-unit](normunit).
     - Bytes per [normalization-unit](normunit)

(SPI)= ### Workgroup manager (SPI)

The workgroup manager (SPI) is the bridge between the `command
processor <CP>`__ and the `compute units <CU>`__. After the `command
processor <cp>`__ processes a kernel dispatch, it will then pass the
dispatch off to the workgroup manager, which then schedules
`workgroups <workgroup>`__ onto the `compute units <CU>`__. As
workgroups complete execution and resources become available, the
workgroup manager will schedule new workgroups onto `compute
units <CU>`__. The workgroup manager’s metrics therefore are focused on
reporting, e.g.:

-  Utilizations of various parts of the accelerator that the workgroup
   manager interacts with (and the workgroup manager itself)
-  How many workgroups were dispatched, their size, and how many
   resources they used
-  Percent of scheduler opportunities (cycles) where workgroups failed
   to dispatch, and
-  Percent of scheduler opportunities (cycles) where workgroups failed
   to dispatch due to lack of a specific resource on the CUs (e.g., too
   many VGPRs allocated)

This gives the user an idea of why the workgroup manager couldn’t
schedule more wavefronts onto the device, and is most useful for
workloads that the user suspects to be scheduling/launch-rate limited.

As discussed in the `command processor <cp>`__ description, the command
processor on AMD Instinct(tm) MI architectures contains four hardware
scheduler-pipes, each with eight software threads (`“Vega10” -
Mantor <https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.120-Radeon-Vega10-Mantor-AMD-f1.pdf>`__,
slide 19). Each scheduler-pipe can issue a kernel dispatch to the
workgroup manager to schedule concurrently. Therefore, some workgroup
manager metrics are presented relative to the utilization of these
scheduler-pipes (e.g., whether all four are issuing concurrently).

.. code:: {note}

   Current versions of the profiling libraries underlying Omniperf attempt to serialize concurrent kernels running on the accelerator, as the performance counters on the device are global (i.e., shared between concurrent kernels).
   This means that these scheduler-pipe utilization metrics are expected to reach e.g., a maximum of one pipe active, i.e., only 25\%.

Workgroup Manager Utilizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section describes the utilization of the workgroup manager, and the
hardware components it interacts with.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Accelerator Utilization
     - The percent of cycles in the kernel where the accelerator was actively doing any work.
     - Percent
   * - Scheduler-Pipe Utilization
     - The percent of [total scheduler-pipe cycles](TotalPipeCycles) in the kernel where the scheduler-pipes were actively doing any work.  Note: this value is expected to range between 0-25%, see note in [workgroup-manager](SPI) description.
     - Percent
   * - Workgroup Manager Utilization
     - The percent of cycles in the kernel where the Workgroup Manager was actively doing any work.
     - Percent
   * - Shader Engine Utilization
     - The percent of [total shader-engine cycles](TotalSECycles) in the kernel where any CU in a shader-engine was actively doing any work, normalized over all shader-engines.  Low values (e.g., << 100%) indicate that the accelerator was not fully saturated by the kernel, or a potential load-imbalance issue.
     - Percent
   * - SIMD Utilization
     - The percent of [total SIMD cycles](TotalSIMDCycles) in the kernel where any [SIMD](VALU) on a CU was actively doing any work, summed over all CUs.  Low values (e.g., << 100%) indicate that the accelerator was not fully saturated by the kernel, or a potential load-imbalance issue.
     - Percent
   * - Dispatched Workgroups
     - The total number of workgroups forming this kernel launch.
     - Workgroups
   * - Dispatched Wavefronts
     - The total number of wavefronts, summed over all workgroups, forming this kernel launch.
     - Wavefronts
   * - VGPR Writes
     - The average number of cycles spent initializing [VGPRs](valu) at wave creation.
     - Cycles/wave
   * - SGPR Writes
     - The average number of cycles spent initializing [SGPRs](salu) at wave creation.
     - Cycles/wave

Workgroup Manager - Resource Allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This panel gives more detail on how workgroups/wavefronts were scheduled
onto compute units, and what occupancy limiters they hit (if any). When
analyzing these metrics, the user should also take into account their
achieved occupancy (i.e., `Wavefront
occupancy <Wavefront_runtime_stats>`__). A kernel may be occupancy
limited by e.g., LDS usage, but may still achieve high occupancy levels
such that improving occupancy further may not improve performance. See
the `Workgroup Manager - Occupancy Limiters <Occupancy_example>`__
example for more details.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - Not-scheduled Rate (Workgroup Manager)
     - The percent of [total scheduler-pipe cycles](TotalPipeCycles) in the kernel where a workgroup could not be scheduled to a [CU](CU) due to a bottleneck within the workgroup manager rather than a lack of a [CU](CU)/[SIMD](VALU) with sufficient resources.  Note: this value is expected to range between 0-25%, see note in [workgroup-manager](SPI) description.
     - Percent
   * - Not-scheduled Rate (Scheduler-Pipe)
     - The percent of [total scheduler-pipe cycles](TotalPipeCycles) in the kernel where a workgroup could not be scheduled to a [CU](CU) due to a bottleneck within the scheduler-pipes rather than a lack of a [CU](CU)/[SIMD](VALU) with sufficient resources.  Note: this value is expected to range between 0-25%, see note in [workgroup-manager](SPI) description.
     - Percent
   * - Scheduler-Pipe Stall Rate
     - The percent of [total scheduler-pipe cycles](TotalPipeCycles) in the kernel where a workgroup could not be scheduled to a [CU](CU) due to occupancy limitations (i.e., a lack of a [CU](CU)/[SIMD](VALU) with sufficient resources).  Note: this value is expected to range between 0-25%, see note in [workgroup-manager](SPI) description.
     - Percent
   * - Scratch Stall Rate
     - The percent of [total shader-engine cycles](TotalSECycles) in the kernel where a workgroup could not be scheduled to a [CU](CU) due to lack of [private (a.k.a., scratch) memory](Mtype) slots.  While this can reach up to 100\%, we note that the actual occupancy limitations on a kernel using private memory are typically quite small (e.g., <1\% of the total number of waves that can be scheduled to an accelerator).
     - Percent
   * - Insufficient SIMD Waveslots
     - The percent of [total SIMD cycles](TotalSIMDCycles) in the kernel where a workgroup could not be scheduled to a [SIMD](valu) due to lack of available [waveslots](valu).
     - Percent
   * - Insufficient SIMD VGPRs
     - The percent of [total SIMD cycles](TotalSIMDCycles) in the kernel where a workgroup could not be scheduled to a [SIMD](valu) due to lack of available [VGPRs](valu).
     - Percent
   * - Insufficient SIMD SGPRs
     - The percent of [total SIMD cycles](TotalSIMDCycles) in the kernel where a workgroup could not be scheduled to a [SIMD](valu) due to lack of available [SGPRs](salu).
     - Percent
   * - Insufficient CU LDS
     - The percent of [total CU cycles](TotalCUCycles) in the kernel where a workgroup could not be scheduled to a [CU](cu) due to lack of available [LDS](lds).
     - Percent
   * - Insufficient CU Barriers
     - The percent of [total CU cycles](TotalCUCycles) in the kernel where a workgroup could not be scheduled to a [CU](cu) due to lack of available [barriers](barrier).
     - Percent
   * - Reached CU Workgroup Limit
     - The percent of [total CU cycles](TotalCUCycles) in the kernel where a workgroup could not be scheduled to a [CU](cu) due to limits within the workgroup manager.  This is expected to be always be zero on CDNA2 or newer accelerators (and small for previous accelerators).
     - Percent
   * - Reached CU Wavefront Limit
     - The percent of [total CU cycles](TotalCUCycles) in the kernel where a wavefront could not be scheduled to a [CU](cu) due to limits within the workgroup manager.  This is expected to be always be zero on CDNA2 or newer accelerators (and small for previous accelerators).
     - Percent

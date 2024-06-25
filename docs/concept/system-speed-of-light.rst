System Speed-of-Light
---------------------

.. code:: {warning}

   The theoretical maximum throughput for some metrics in this section are currently computed with the maximum achievable clock frequency, as reported by `rocminfo`, for an accelerator.  This may not be realistic for all workloads.

   In addition, not all metrics (e.g., FLOP counters) are available on all AMD Instinct(tm) MI accelerators.
   For more detail on how operations are counted, see the [FLOP counting convention](FLOP_count) section.

Finally, the system speed-of-light summarizes some of the key metrics
from various sections of Omniperf’s profiling report.

.. code:: {list-table}

   :header-rows: 1
   :widths: 20 65 15
   :class: noscroll-table
   * - Metric
     - Description
     - Unit
   * - [VALU](valu) FLOPs
     - The total floating-point operations executed per second on the [VALU](valu).  This is also presented as a percent of the peak theoretical FLOPs achievable on the specific accelerator. Note: this does not include any floating-point operations from [MFMA](mfma) instructions.
     - GFLOPs
   * - [VALU](valu) IOPs
     - The total integer operations executed per second on the [VALU](valu).  This is also presented as a percent of the peak theoretical IOPs achievable on the specific accelerator. Note: this does not include any integer operations from [MFMA](mfma) instructions.
     - GIOPs
   * - [MFMA](mfma) FLOPs (BF16)
     - The total number of 16-bit brain floating point [MFMA](mfma) operations executed per second. Note: this does not include any 16-bit brain floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical BF16 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - [MFMA](mfma) FLOPs (F16)
     - The total number of 16-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 16-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F16 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - [MFMA](mfma) FLOPs (F32)
     - The total number of 32-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 32-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F32 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - [MFMA](mfma) FLOPs (F64)
     - The total number of 64-bit floating point [MFMA](mfma) operations executed per second. Note: this does not include any 64-bit floating point operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical F64 MFMA operations achievable on the specific accelerator.
     - GFLOPs
   * - [MFMA](mfma) IOPs (INT8)
     - The total number of 8-bit integer [MFMA](mfma) operations executed per second. Note: this does not include any 8-bit integer operations from [VALU](valu) instructions. This is also presented as a percent of the peak theoretical INT8 MFMA operations achievable on the specific accelerator.
     - GIOPs
   * - [SALU](salu) Utilization
     - Indicates what percent of the kernel's duration the [SALU](salu) was busy executing instructions.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [SALU](salu) / [SMEM](salu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - [VALU](valu) Utilization
     - Indicates what percent of the kernel's duration the [VALU](valu) was busy executing instructions.  Does not include [VMEM](valu) operations.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [VALU](valu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - [MFMA](mfma) Utilization
     - Indicates what percent of the kernel's duration the [MFMA](mfma) unit was busy executing instructions.  Computed as the ratio of the total number of cycles the [MFMA](mfma) was busy over the [total CU cycles](TotalCUCycles).
     - Percent
   * - [VMEM](valu) Utilization
     - Indicates what percent of the kernel's duration the [VMEM](valu) unit was busy executing instructions, including both global/generic and spill/scratch operations (see the [VMEM instruction count metrics](TA_inst) for more detail).  Does not include [VALU](valu) operations.  Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [VMEM](valu) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - [Branch](branch) Utilization
     - Indicates what percent of the kernel's duration the [Branch](branch) unit was busy executing instructions. Computed as the ratio of the total number of cycles spent by the [scheduler](scheduler) issuing [Branch](branch) instructions over the [total CU cycles](TotalCUCycles).
     - Percent
   * - [VALU](valu) Active Threads
     - Indicates the average level of [divergence](Divergence) within a wavefront over the lifetime of the kernel. The number of work-items that were active in a wavefront during execution of each [VALU](valu) instruction, time-averaged over all VALU instructions run on all wavefronts in the kernel.
     - Work-items
   * - IPC
     - The ratio of the total number of instructions executed on the [CU](cu) over the [total active CU cycles](TotalActiveCUCycles). This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator.
     - Instructions per-cycle
   * - Wavefront Occupancy
     - The time-averaged number of wavefronts resident on the accelerator over the lifetime of the kernel. Note: this metric may be inaccurate for short-running kernels (<< 1ms).   This is also presented as a percent of the peak theoretical occupancy achievable on the specific accelerator.
     - Wavefronts
   * - [LDS](lds) Theoretical Bandwidth
     - Indicates the maximum amount of bytes that could have been loaded from/stored to/atomically updated in the LDS per unit time (see [LDS Bandwidth](lds_bandwidth) example for more detail).  This is also presented as a percent of the peak theoretical F64 MFMA operations achievable on the specific accelerator.
     - GB/s
   * - [LDS](lds) Bank Conflicts/Access
     - The ratio of the number of cycles spent in the [LDS scheduler](lds) due to bank conflicts (as determined by the conflict resolution hardware) to the base number of cycles that would be spent in the LDS scheduler in a completely uncontended case.  This is also presented in normalized form (i.e., the Bank Conflict Rate).
     - Conflicts/Access
   * - [vL1D](vL1D) Cache Hit Rate
     - The ratio of the number of vL1D cache line requests that hit in vL1D cache over the total number of cache line requests to the [vL1D Cache RAM](TC).
     - Percent
   * - [vL1D](vL1D) Cache Bandwidth
     - The number of bytes looked up in the vL1D cache as a result of [VMEM](VALU) instructions per unit time.  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.  This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator.
     - GB/s
   * - [L2](L2) Cache Hit Rate
     - The ratio of the number of L2 cache line requests that hit in the L2 cache over the total number of incoming cache line requests to the L2 cache.
     - Percent
   * - [L2](L2) Cache Bandwidth
     - The number of bytes looked up in the L2 cache per unit time.  The number of bytes is calculated as the number of cache lines requested multiplied by the cache line size.  This value does not consider partial requests, so e.g., if only a single value is requested in a cache line, the data movement will still be counted as a full cache line.  This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator.
     - GB/s
   * - [L2](L2)-Fabric Read BW
     - The number of bytes read by the L2 over the [Infinity Fabric(tm) interface](l2fabric) per unit time. This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator.
     - GB/s
   * - [L2](L2)-Fabric Write and Atomic BW
     - The number of bytes sent by the L2 over the [Infinity Fabric(tm) interface](l2fabric) by write and atomic operations per unit time. This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator.
     - GB/s
   * - [L2](L2)-Fabric Read Latency
     - The time-averaged number of cycles read requests spent in Infinity Fabric(tm) before data was returned to the L2.
     - Cycles
   * - [L2](L2)-Fabric Write Latency
     - The time-averaged number of cycles write requests spent in Infinity Fabric(tm) before a completion acknowledgement was returned to the L2.
     - Cycles
   * - [sL1D](sL1D) Cache Hit Rate
     - The percent of sL1D requests that hit on a previously loaded line the cache. Calculated as the ratio of the number of sL1D requests that hit over the number of all sL1D requests.
     - Percent
   * - [sL1D](sL1D) Bandwidth
     - The number of bytes looked up in the sL1D cache per unit time. This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator. 
     - GB/s
   * - [L1I](L1I) Bandwidth
     - The number of bytes looked up in the L1I cache per unit time. This is also presented as a percent of the peak theoretical bandwidth achievable on the specific accelerator. 
     - GB/s
   * - [L1I](L1I) Cache Hit Rate
     - The percent of L1I requests that hit on a previously loaded line the cache. Calculated as the ratio of the number of L1I requests that hit over the number of all L1I requests.
     - Percent
   * - [L1I](L1I) Fetch Latency
     - The average number of cycles spent to fetch instructions to a [CU](cu).
     - Cycles

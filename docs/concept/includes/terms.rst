.. list-table::
   :header-rows: 1

   * - Name
     - Description
     - Unit

   * - Kernel time
     - The number of seconds the accelerator was executing a kernel, from the
       :ref:`command processor <def-cp>`'s (CP) start-of-kernel
       timestamp (a number of cycles after the CP beings processing the packet)
       to the CP's end-of-kernel timestamp (a number of cycles before the CP
       stops processing the packet).
     - Seconds

   * - Kernel cycles
     - The number of cycles the accelerator was active doing *any* work, as
       measured by the :ref:`command processor <def-cp>` (CP).
     - Cycles

   * - Total CU cycles
     - The number of cycles the accelerator was active doing *any* work
       (that is, kernel cycles), multiplied by the number of
       :ref:`compute units <def-cu>` on the accelerator. A
       measure of the total possible active cycles the compute units could be
       doing work, useful for the normalization of metrics inside the CU.
     - Cycles

   * - Total active CU cycles
     - The number of cycles a CU on the accelerator was active doing *any*
       work, summed over all :ref:`compute units <def-cu>` on the
       accelerator.
     - Cycles

   * - Total SIMD cycles
     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`SIMDs <def-cu>` on the accelerator. A measure of the
       total possible active cycles the SIMDs could be doing work, useful for
       the normalization of metrics inside the CU.
     - Cycles

   * - Total L2 cycles
     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of :ref:`L2 <def-l2>`
       channels on the accelerator. A measure of the total possible active
       cycles the L2 channels could be doing work, useful for the normalization
       of metrics inside the L2.
     - Cycles

   * - Total active L2 cycles
     - The number of cycles a channel of the L2 cache was active doing *any*
       work, summed over all :ref:`L2 <def-l2>` channels on the accelerator.
     - Cycles

   * - Total sL1D cycles
     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`scalar L1 data caches <def-sl1d>` on the accelerator. A measure of
       the total possible active cycles the sL1Ds could be doing work, useful
       for the normalization of metrics inside the sL1D.
     - Cycles

   * - Total L1I cycles
     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`L1 instruction caches <def-l1i>` (L1I) on the accelerator. A
       measure of the total possible active cycles the L1Is could be doing
       work, useful for the normalization of metrics inside the L1I.
     - Cycles

   * - Total scheduler-pipe cycles
     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`scheduler pipes <def-cp>` on the accelerator. A measure of the
       total possible active cycles the scheduler-pipes could be doing work,
       useful for the normalization of metrics inside the
       :ref:`workgroup manager <def-spi>` and :ref:`command processor <def-cp>`.
     - Cycles

   * - Total shader-engine cycles
     - The total number of cycles the accelerator was active doing *any* work,
       multiplied by the number of :ref:`shader engines <def-se>` on the
       accelerator. A measure of the total possible active cycles the shader
       engines could be doing work, useful for the normalization of
       metrics inside the :ref:`workgroup manager <def-spi>`.
     - Cycles

   * - Thread-requests
     - The number of unique memory addresses accessed by a single memory
       instruction. On AMD Instinct accelerators, this has a maximum of 64
       (that is, the size of the :ref:`wavefront <def-wavefront>`).
     - Addresses

   * - Work-item
     - A single *thread*, or lane, of execution that executes in lockstep with
       the rest of the work-items comprising a :ref:`wavefront <def-wavefront>`
       of execution.
     - N/A

   * - Wavefront
     - A group of work-items, or threads, that execute in lockstep on the
       :ref:`compute unit <def-cu>`. On AMD Instinct accelerators, the
       wavefront size is always 64 work-items.
     - N/A

   * - Workgroup
     - A group of wavefronts that execute on the same
       :ref:`compute unit <def-cu>`, and can cooperatively execute and share
       data via the use of synchronization primitives, :ref:`LDS <def-lds>`,
       atomics, and others.
     - N/A

   * - Divergence
     - Divergence within a wavefront occurs when not all work-items are active
       when executing an instruction, that is, due to non-uniform control flow
       within a wavefront. Can reduce execution efficiency by causing,
       for instance, the :ref:`VALU <def-valu>` to need to execute both
       branches of a conditional with different sets of work-items active.
     - N/A

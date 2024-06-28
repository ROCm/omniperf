**********************
Command processor (CP)
**********************

The command processor (CP) is responsible for interacting with the AMDGPU Kernel
Driver (a.k.a., the Linux Kernel) on the CPU and
for interacting with user-space HSA clients when they submit commands to
HSA queues. Basic tasks of the CP include reading commands (e.g.,
corresponding to a kernel launch) out of `HSA
Queues <http://hsafoundation.com/wp-content/uploads/2021/02/HSA-Runtime-1.2.pdf>`__
(Sec. 2.5), scheduling work to subsequent parts of the scheduler
pipeline, and marking kernels complete for synchronization events on the
host.

The command processor is composed of two sub-components:

-  Fetcher (CPF): Fetches commands out of memory to hand them over to
   the CPC for processing
-  Packet Processor (CPC): The micro-controller running the command
   processing firmware that decodes the fetched commands, and (for
   kernels) passes them to the `Workgroup Processors <SPI>`__ for
   scheduling

Before scheduling work to the accelerator, the command-processor can
first acquire a memory fence to ensure system consistency `(Sec
2.6.4) <http://hsafoundation.com/wp-content/uploads/2021/02/HSA-Runtime-1.2.pdf>`__.
After the work is complete, the command-processor can apply a
memory-release fence. Depending on the AMD CDNA accelerator under
question, either of these operations *may* initiate a cache write-back
or invalidation.

Analyzing command processor performance is most interesting for kernels
that the user suspects to be scheduling/launch-rate limited. The command
processor’s metrics therefore are focused on reporting, e.g.:

-  Utilization of the fetcher
-  Utilization of the packet processor, and decoding processing packets
-  Fetch/processing stalls

Command Processor Fetcher (CPF) metrics
=======================================

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric
     - Description
     - Unit
   * - CPF Utilization
     - Percent of total cycles where the CPF was busy actively doing any work.  The ratio of CPF busy cycles over total cycles counted by the CPF.
     - Percent
   * - CPF Stall
     - Percent of CPF busy cycles where the CPF was stalled for any reason.
     - Percent
   * - CPF-L2 Utilization
     - Percent of total cycles counted by the CPF-[L2](L2) interface where the CPF-L2 interface was active doing any work.  The ratio of CPF-L2 busy cycles over total cycles counted by the CPF-L2.
     - Percent
   * - CPF-L2 Stall
     - Percent of CPF-L2 busy cycles where the CPF-[L2](L2) interface was stalled for any reason.
     - Percent
   * - CPF-UTCL1 Stall
     - Percent of CPF busy cycles where the CPF was stalled by address translation. 
     - Percent

Command Processor Packet Processor (CPC) metrics
================================================

.. list-table::
   :header-rows: 1
   :widths: 20 65 15

   * - Metric
     - Description
     - Unit
   * - CPC Utilization
     - Percent of total cycles where the CPC was busy actively doing any work. The ratio of CPC busy cycles over total cycles counted by the CPC.
     - Percent
   * - CPC Stall
     - Percent of CPC busy cycles where the CPC was stalled for any reason.
     - Percent
   * - CPC Packet Decoding Utilization
     - Percent of CPC busy cycles spent decoding commands for processing.
     - Percent
   * - CPC-Workgroup Manager Utilization
     - Percent of CPC busy cycles spent dispatching workgroups to the [Workgroup Manager](SPI).
     - Percent
   * - CPC-L2 Utilization
     - Percent of total cycles counted by the CPC-[L2](L2) interface where the CPC-L2 interface was active doing any work.
     - Percent
   * - CPC-UTCL1 Stall
     - Percent of CPC busy cycles where the CPC was stalled by address translation.
     - Percent
   * - CPC-UTCL2 Utilization
     - Percent of total cycles counted by the CPC's L2 address translation interface where the CPC was busy doing address translation work.
     - Percent

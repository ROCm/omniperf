.. _vmem-example:

Vector memory operation counting
================================

Global / Generic (FLAT)
-----------------------

For this example, consider the
:dev-sample:`vector memory sample <vmem.hip>` distributed as a part of
Omniperf. This code launches many different
versions of a simple read/write/atomic-only kernels targeting various
address spaces, e.g. below is our simple ``global_write`` kernel:

.. code:: cpp

   // write to a global pointer
   __global__ void global_write(int* ptr, int zero) {
     ptr[threadIdx.x] = zero;
   }

This example was compiled and run on an MI250 accelerator using ROCm
v5.6.0, and Omniperf v2.0.0.

.. code:: shell-session

   $ hipcc -O3 --save-temps vmem.hip -o vmem

We have also chosen to include the ``--save-temps`` flag to save the
compiler temporary files, such as the generated CDNA assembly code, for
inspection.

Finally, we generate our omniperf profile as:

.. code:: shell-session

   $ omniperf profile -n vmem --no-roof -- ./vmem

(Flat_design)= #### Design note

We should explain some of the more peculiar line(s) of code in our
example, e.g., the use of compiler builtins and explicit address space
casting, etc.

.. code:: cpp

   // write to a generic pointer
   typedef int __attribute__((address_space(0)))* generic_ptr;

   __attribute__((noinline)) __device__ void generic_store(generic_ptr ptr, int zero) { *ptr = zero; }

   __global__ void generic_write(int* ptr, int zero, int filter) {
     __shared__ int lds[1024];
     int* generic = (threadIdx.x < filter) ? &ptr[threadIdx.x] : &lds[threadIdx.x];
     generic_store((generic_ptr)generic, zero);
   }

One of our aims in this example is to demonstrate the use of the
`‘generic’ (a.k.a.,
FLAT) <https://llvm.org/docs/AMDGPUUsage.html#address-space-identifier>`__
address space. This address space is typically used when the compiler
cannot statically prove where the backing memory is located.

To try to *force* the compiler to use this address space, we have
applied ``__attribute__((noinline))`` to the ``generic_store`` function
to have the compiler treat it as a function call (i.e., on the
other-side of which, the address space may not be known). However, in a
trivial example such as this, the compiler may choose to specialize the
``generic_store`` function to the two address spaces that may provably
be used from our translation-unit, i.e., `‘local’ (a.k.a.,
LDS) <Mspace>`__ and `‘global’ <Mspace>`__. Hence, we forcibly cast the
address space to `‘generic’ (i.e., FLAT) <Mspace>`__ to avoid this
compiler optimization.

.. code:: {warning}

   While convenient for our example here, this sort of explicit address space casting can lead to strange compilation errors, and in the worst cases, incorrect results and thus use is discouraged in production code.

For more details on address spaces, the reader is referred to the
`address-space section <Mspace>`__.

Global Write
-------------

First, we demonstrate our simple ``global_write`` kernel:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 1 -b 10.3 15.1.4 15.1.5 15.1.6 15.1.7 15.1.8 15.1.9 15.1.10 15.1.11  -n per_kernel
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                          │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ global_write(int*, int) [clone .kd] │    1.00 │   2400.00 │    2400.00 │      2400.00 │ 100.00 │
   ╘════╧═════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   15. Address Processing Unit and Data Return Path (TA/TD)
   15.1 Address Processing Unit
   ╒═════════╤═════════════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                      │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═════════════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 15.1.4  │ Total Instructions          │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.5  │ Global/Generic Instr        │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.6  │ Global/Generic Read Instr   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.7  │ Global/Generic Write Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.8  │ Global/Generic Atomic Instr │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.9  │ Spill/Stack Instr           │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.10 │ Spill/Stack Read Instr      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼─────────────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 15.1.11 │ Spill/Stack Write Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═════════════════════════════╧═══════╧═══════╧═══════╧══════════════════╛

Here, we have presented both the information in the VMEM Instruction Mix
table (10.3) and the Address Processing Unit (15.1). We note that this
data is expected to be identical, and hence we omit table 15.1 in our
subsequent examples.

In addition, as expected, we see a single Global/Generic write
instruction (10.3.2, 15.1.7). Inspecting the generated assembly:

.. code:: asm

           .protected      _Z12global_writePii     ; -- Begin function _Z12global_writePii
           .globl  _Z12global_writePii
           .p2align        8
           .type   _Z12global_writePii,@function
   _Z12global_writePii:                    ; @_Z12global_writePii
   ; %bb.0:
           s_load_dword s2, s[4:5], 0x8
           s_load_dwordx2 s[0:1], s[4:5], 0x0
           v_lshlrev_b32_e32 v0, 2, v0
           s_waitcnt lgkmcnt(0)
           v_mov_b32_e32 v1, s2
           global_store_dword v0, v1, s[0:1]
           s_endpgm
           .section        .rodata,#alloc
           .p2align        6, 0x0
           .amdhsa_kernel _Z12global_writePii

we see that this corresponds to an instance of a ``global_store_dword``
operation.

.. code:: {note}

   The assembly in these experiments were generated for an [MI2XX](2xxnote) accelerator using ROCm 5.6.0, and may change depending on ROCm versions and the targeted hardware architecture

(Generic_write)= #### Generic Write to LDS

Next, we examine a generic write. As discussed
`previously <Flat_design>`__, our ``generic_write`` kernel uses an
address space cast to *force* the compiler to choose our desired address
space, regardless of other optimizations that may be possible.

We also note that the ``filter`` parameter passed in as a kernel
argument (see
`example <https://github.com/ROCm/omniperf/blob/dev/sample/vmem.hip>`__,
or `design note <Flat_design>`__) is set to zero on the host, such that
we always write to the ‘local’ (LDS) memory allocation ``lds``.

Examining this kernel in the VMEM Instruction Mix table yields:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 2 -b 10.3 -n per_kernel
   <...>
   0. Top Stat
   ╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                               │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ generic_write(int*, int, int) [clone .kd │    1.00 │   2880.00 │    2880.00 │      2880.00 │ 100.00 │
   │    │ ]                                        │         │           │            │              │        │
   ╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛

As expected we see a single generic write (10.3.2). In the assembly
generated for this kernel (in particular, we care about the
``generic_store`` function). We see that this corresponds to a
``flat_store_dword`` instruction:

.. code:: asm

           .type   _Z13generic_storePii,@function
   _Z13generic_storePii:                   ; @_Z13generic_storePii
   ; %bb.0:
           s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
           flat_store_dword v[0:1], v2
           s_waitcnt vmcnt(0) lgkmcnt(0)
           s_setpc_b64 s[30:31]
   .Lfunc_end0:

In addition, we note that we can observe the destination of this request
by looking at the LDS Instructions metric (12.2.0):

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 2 -b 12.2.0 -n per_kernel
   <...>
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric     │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 12.2.0  │ LDS Instrs │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ╘═════════╧════════════╧═══════╧═══════╧═══════╧══════════════════╛

which indicates one LDS access.

.. code:: {note}

   Exercise for the reader: if this access had been targeted at global memory (e.g., by changing value of `filter`), where should we look for the memory traffic?  Hint: see our [generic read](Generic_read) example.

Global read
^^^^^^^^^^^

Next, we examine a simple global read operation:

.. code:: cpp

   __global__ void global_read(int* ptr, int zero) {
     int x = ptr[threadIdx.x];
     if (x != zero) {
       ptr[threadIdx.x] = x + 1;
     }
   }

Here we observe a now familiar pattern: - Read a value in from global
memory - Have a write hidden behind a conditional that is impossible for
the compiler to statically eliminate, but is identically false. In this
case, our ``main()`` function initializes the data in ``ptr`` to zero.

Running Omniperf on this kernel yields:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 3 -b 10.3 -n per_kernel
   <...>
   0. Top Stat
   ╒════╤════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                         │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ global_read(int*, int) [clone .kd] │    1.00 │   4480.00 │    4480.00 │      4480.00 │ 100.00 │
   ╘════╧════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛

Here we see a single global/generic instruction (10.3.0) which, as
expected, is a read (10.3.1).

(Generic_read)= #### Generic read from global memory

For our generic read example, we choose to change our target for the
generic read to be global memory:

.. code:: cpp

   __global__ void generic_read(int* ptr, int zero, int filter) {
     __shared__ int lds[1024];
     if (static_cast<int>(filter - 1) == zero) {
       lds[threadIdx.x] = 0; // initialize to zero to avoid conditional, but hide behind _another_ conditional
     }
     int* generic;
     if (static_cast<int>(threadIdx.x) > filter - 1) {
       generic = &ptr[threadIdx.x];
     } else {
       generic = &lds[threadIdx.x];
       abort();
     }
     int x = generic_load((generic_ptr)generic);
     if (x != zero) {
       ptr[threadIdx.x] = x + 1;
     }
   }

In addition to our usual ``if (condition_that_wont_happen)`` guard
around the write operation, there is an additional conditional around
the initialization of the ``lds`` buffer. We note that it’s typically
required to write to this buffer to prevent the compiler from
eliminating the local memory branch entirely due to undefined behavior
(use of an uninitialized value). However, to report *only* our global
memory read, we again hide this initialization behind an identically
false conditional (both ``zero`` and ``filter`` are set to zero in the
kernel launch). Note that this is a *different* conditional from our
pointer assignment (to avoid combination of the two).

Running Omniperf on this kernel reports:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 4 -b 10.3 12.2.0 16.3.10 -n per_kernel
   <...>
   0. Top Stat
   ╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                               │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ generic_read(int*, int, int) [clone .kd] │    1.00 │   2240.00 │    2240.00 │      2240.00 │ 100.00 │
   ╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric     │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 12.2.0  │ LDS Instrs │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   16. Vector L1 Data Cache
   16.3 L1D Cache Accesses
   ╒═════════╤════════════╤═══════╤═══════╤═══════╤════════════════╕
   │ Index   │ Metric     │   Avg │   Min │   Max │ Unit           │
   ╞═════════╪════════════╪═══════╪═══════╪═══════╪════════════════╡
   │ 16.3.10 │ L1-L2 Read │  1.00 │  1.00 │  1.00 │ Req per kernel │
   ╘═════════╧════════════╧═══════╧═══════╧═══════╧════════════════╛

Here we observe: - A single global/generic read operation (10.3.1),
which - Is not an LDS instruction (12.2), as seen in our `generic
write <Generic_write>`__ example, but is instead - An L1-L2 read
operation (16.3.10)

That is, we have successfully targeted our generic read at global
memory. Inspecting the assembly shows this corresponds to a
``flat_load_dword`` instruction.

(Global_atomic)= #### Global atomic

Our global atomic kernel:

.. code:: cpp

   __global__ void global_atomic(int* ptr, int zero) {
     atomicAdd(ptr, zero);
   }

simply atomically adds a (non-compile-time) zero value to a pointer.

Running Omniperf on this kernel yields:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 5 -b 10.3 16.3.12 -n per_kernel
   <...>
   0. Top Stat
   ╒════╤══════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                           │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪══════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ global_atomic(int*, int) [clone .kd] │    1.00 │   4640.00 │    4640.00 │      4640.00 │ 100.00 │
   ╘════╧══════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   16. Vector L1 Data Cache
   16.3 L1D Cache Accesses
   ╒═════════╤══════════════╤═══════╤═══════╤═══════╤════════════════╕
   │ Index   │ Metric       │   Avg │   Min │   Max │ Unit           │
   ╞═════════╪══════════════╪═══════╪═══════╪═══════╪════════════════╡
   │ 16.3.12 │ L1-L2 Atomic │  1.00 │  1.00 │  1.00 │ Req per kernel │
   ╘═════════╧══════════════╧═══════╧═══════╧═══════╧════════════════╛

Here we see a single global/generic atomic instruction (10.3.3), which
corresponds to an L1-L2 atomic request (16.3.12).

(Generic_atomic)= #### Generic, mixed atomic

In our final global/generic example, we look at a case where our generic
operation targets both LDS and global memory:

.. code:: cpp

   __global__ void generic_atomic(int* ptr, int filter, int zero) {
     __shared__ int lds[1024];
     int* generic = (threadIdx.x % 2 == filter) ? &ptr[threadIdx.x] : &lds[threadIdx.x];
     generic_atomic((generic_ptr)generic, zero);
   }

This assigns every other work-item to atomically update global memory or
local memory.

Running this kernel through Omniperf shows:

.. code:: shell-session

   $ omniperf analyze -p workloads/vmem/mi200/ --dispatch 6 -b 10.3 12.2.0 16.3.12 -n per_kernel
   <...>
   0. Top Stat
   ╒════╤══════════════════════════════════════════╤═════════╤═══════════╤════════════╤══════════════╤════════╕
   │    │ KernelName                               │   Count │   Sum(ns) │   Mean(ns) │   Median(ns) │    Pct │
   ╞════╪══════════════════════════════════════════╪═════════╪═══════════╪════════════╪══════════════╪════════╡
   │  0 │ generic_atomic(int*, int, int) [clone .k │    1.00 │   3360.00 │    3360.00 │      3360.00 │ 100.00 │
   │    │ d]                                       │         │           │            │              │        │
   ╘════╧══════════════════════════════════════════╧═════════╧═══════════╧════════════╧══════════════╧════════╛


   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric     │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 12.2.0  │ LDS Instrs │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ╘═════════╧════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   16. Vector L1 Data Cache
   16.3 L1D Cache Accesses
   ╒═════════╤══════════════╤═══════╤═══════╤═══════╤════════════════╕
   │ Index   │ Metric       │   Avg │   Min │   Max │ Unit           │
   ╞═════════╪══════════════╪═══════╪═══════╪═══════╪════════════════╡
   │ 16.3.12 │ L1-L2 Atomic │  1.00 │  1.00 │  1.00 │ Req per kernel │
   ╘═════════╧══════════════╧═══════╧═══════╧═══════╧════════════════╛

That is, we see: - A single generic atomic instruction (10.3.3) that
maps to both - an LDS instruction (12.2.0), and - an L1-L2 atomic
request (16.3)

We have demonstrated the ability of the generic address space to
*dynamically* target different backing memory!

(buffermembench)= ### Spill/Scratch (BUFFER)

Next we examine the use of ‘Spill/Scratch’ memory. On current CDNA
accelerators such as the `MI2XX <2xxnote>`__, this is implemented using
the `private <mspace>`__ memory space, which maps to `‘scratch’
memory <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces>`__
in AMDGPU hardware terminology. This type of memory can be accessed via
different instructions depending on the specific architecture targeted.
However, current CDNA accelerators such as the `MI2XX <2xxnote>`__ use
so called ``buffer`` instructions to access private memory in a simple
(and typically) coalesced manner. See `Sec. 9.1, ‘Vector Memory Buffer
Instructions’ of the CDNA2 ISA
guide <https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf>`__
for further reading on this instruction type.

We develop a `simple
kernel <https://github.com/ROCm/omniperf/blob/dev/sample/stack.hip>`__
that uses stack memory:

.. code:: cpp

   #include <hip/hip_runtime.h>
   __global__ void knl(int* out, int filter) {
     int x[1024];
     x[filter] = 0;
     if (threadIdx.x < filter)
       out[threadIdx.x] = x[threadIdx.x];
   }

Our strategy here is to: - Create a large stack buffer (that cannot
reasonably fit into registers) - Write to a compile-time unknown
location on the stack, and then - Behind the typical compile-time
unknown ``if(condition_that_wont_happen)`` - Read from a different,
compile-time unknown, location on the stack and write to global memory
to prevent the compiler from optimizing it out.

This example was compiled and run on an MI250 accelerator using ROCm
v5.6.0, and Omniperf v2.0.0.

.. code:: shell-session

   $ hipcc -O3 stack.hip -o stack.hip

and profiled using omniperf:

.. code:: shell-session

   $ omniperf profile -n stack --no-roof -- ./stack
   <...>
   $ omniperf analyze -p workloads/stack/mi200/  -b 10.3 16.3.11 -n per_kernel
   <...>
   10. Compute Units - Instruction Mix
   10.3 VMEM Instr Mix
   ╒═════════╤═══════════════════════╤═══════╤═══════╤═══════╤══════════════════╕
   │ Index   │ Metric                │   Avg │   Min │   Max │ Unit             │
   ╞═════════╪═══════════════════════╪═══════╪═══════╪═══════╪══════════════════╡
   │ 10.3.0  │ Global/Generic Instr  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.1  │ Global/Generic Read   │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.2  │ Global/Generic Write  │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.3  │ Global/Generic Atomic │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.4  │ Spill/Stack Instr     │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.5  │ Spill/Stack Read      │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.6  │ Spill/Stack Write     │  1.00 │  1.00 │  1.00 │ Instr per kernel │
   ├─────────┼───────────────────────┼───────┼───────┼───────┼──────────────────┤
   │ 10.3.7  │ Spill/Stack Atomic    │  0.00 │  0.00 │  0.00 │ Instr per kernel │
   ╘═════════╧═══════════════════════╧═══════╧═══════╧═══════╧══════════════════╛


   --------------------------------------------------------------------------------
   16. Vector L1 Data Cache
   16.3 L1D Cache Accesses
   ╒═════════╤═════════════╤═══════╤═══════╤═══════╤════════════════╕
   │ Index   │ Metric      │   Avg │   Min │   Max │ Unit           │
   ╞═════════╪═════════════╪═══════╪═══════╪═══════╪════════════════╡
   │ 16.3.11 │ L1-L2 Write │  1.00 │  1.00 │  1.00 │ Req per kernel │
   ╘═════════╧═════════════╧═══════╧═══════╧═══════╧════════════════╛

Here we see a single write to the stack (10.3.6), which corresponds to
an L1-L2 write request (16.3.11), i.e., the stack is backed by global
memory and travels through the same memory hierarchy.

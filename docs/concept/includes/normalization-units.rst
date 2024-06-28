.. _normalization-units:

Normalization units
===================

A user-configurable unit by which you can choose to normalize data. Options
include:

.. list-table::
   :header-rows: 1

   * - Name
     - Description

   * - ``per_cycle``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the
       :ref:`kernel cycles <def-kernel-cycles>`, that is, the total number of
       cycles the kernel executed as measured by the
       :ref:`command processor <def-cp>`.

   * - ``per_wave``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the total number of
       :ref:`wavefronts <def-wavefront>` launched in the kernel.

   * - ``per_kernel``
     - The total value of the measured counter or metric that occurred per
       kernel invocation.

   * - ``per_second``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the :ref:`kernel time <def-kernel-time>`,
       that is, the total runtime of the kernel in seconds, as measured by the
       :ref:`command processor <def-cp>`.

By default, Omniperf uses the ``per_wave`` normalization.

The ideal normalization varies depending on your use case. For instance, a
``per_second`` normalization might be useful for FLOP or bandwidth
comparisons, while a ``per_wave`` normalization could be useful to see how many
(and what types) of instructions are used per wavefront; a ``per_kernel``
normalization may be useful to get the total aggregate values of metrics for
comparison between different configurations.

.. meta::
   :description: Omniperf basic usage documentation.
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD,
              basics, usage, operations

***********
Basic usage
***********

The following section outlines basic Omniperf workflows, modes, options, and
operations.

Command line profiler
=====================

Launch and profile the target application using the command line profiler.

The command line profiler launches the target application, calls the
ROCProfiler API via the ``rocprof`` binary, and collects profile results for
the specified kernels, dispatches, and hardware components. If not
specified, Omniperf defaults to collecting all available counters for all
kernels and dispatches launched by the your executable.

To collect the default set of data for all kernels in the target
application, launch, for example:

.. code-block:: shell

   $ omniperf profile -n vcopy_data -- ./vcopy -n 1048576 -b 256

This runs the app, launches each kernel, and generates profiling results. By
default, results are written to a subdirectory with your accelerator's name;
for example, ``./workloads/vcopy_data/MI200/``, where name is configurable
via the ``-n`` argument.

.. note::

   To collect all requested profile information, Omniperf might replay kernels
   multiple times.

.. _basic-filter-data-collection:

Customize data collection
-------------------------

Options are available to specify for which kernels and metrics data should be
collected. Note that you can apply filtering in either the profiling or
analysis stage. Filtering at profiling collection often speeds up your
aggregate profiling run time.

Common filters to customize data collection include:

``-k``, ``--kernel``
   Enables filtering kernels by name.

``-d``, ``--dispatch``
   Enables filtering based on dispatch ID.

``-b``, ``--block``
   Enables collection metrics for only the specified (one or more) hardware
   component blocks.

See :ref:`Filtering <profiling-filtering>` for an in-depth walkthrough.

To view available metrics by hardware block, use the ``--list-metrics``
argument:

.. code-block:: shell

   $ omniperf analyze --list-metrics <sys_arch>

.. _basic-analyze-cli:

Analyze at the command line
---------------------------

After generating a local output folder (for example,
``./workloads/vcopy_data/MI200``), use the command line tool to quickly
interface with profiling results. View different metrics derived from your
profiled results and get immediate access all metrics organized by hardware
blocks.

If you don't apply kernel, dispatch, or hardware block filters at this stage,
analysis is reflective of the entirety of the profiling data.

To interact with profiling results from a different session, provide the
workload path.

``-p``, ``--path``
   Enables you to analyze existing profiling data in the Omniperf CLI.

.. _basic-analyze-grafana:

Analyze in the Grafana GUI
--------------------------

To conduct a more in-depth analysis of profiling results, it's suggested to use
a Grafana GUI with Omniperf. To interact with profiling results, import your
data to the MongoDB instance included in the Omniperf Dockerfile. See
:doc:`../install/grafana-setup`.

To interact with Grafana data, stored in the Omniperf database, enter
``database`` :ref:`mode <modes>`; for example:

.. code-block:: shell

   $ omniperf database --import [CONNECTION OPTIONS]

.. include:: ./includes/modes.rst

.. include:: ./includes/global-options.rst

.. note::

   Omniperf also recognizes the project variable, ``OMNIPERF_COLOR`` should you
   choose to disable colorful output. To disable default colorful behavior, set
   this variable to ``0``.

.. _basic-operations:

Basic operations
================

The following table lists Omniperf's basic operations, their modes, and required
arguments.

.. list-table::
   :header-rows: 1

   * - Operation description
     - Mode
     - Required arguments

   * - Profile a workload
     - ``profile``
     - ``--name``, ``-- <profile_cmd>``

   * - Standalone roofline analysis
     - ``profile``
     - ``--name``, ``--roof-only``, ``-- <profile_cmd>``

   * - Import a workload to database
     - ``database``
     - ``--import``, ``--host``, ``--username``, ``--workload``, ``--team``

   * - Remove a workload from database
     - ``database``
     - ``--remove``, ``--host``, ``--username``, ``--workload``, ``--team``

   * - Launch standalone GUI from CLI
     - ``analyze``
     - ``--path``, ``--gui``

   * - Interact with profiling results from CLI
     - ``analyze``
     - ``--path``

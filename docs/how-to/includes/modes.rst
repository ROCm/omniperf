.. _modes:

Modes
=====

Modes change the fundamental behavior of the Omniperf command line tool.
Depending on which mode you choose, different command line options become
available.

.. _modes-profile:

Profile mode
------------

``profile``
   Launches the target application on the local system using
   :doc:`ROCProfiler <rocprofiler:index>`. Depending on the profiling options
   chosen, selected kernels, dispatches, and or hardware components used by the
   application are profiled. It stores results locally in an output folder:
   ``./workloads/\<name>``.

   .. code-block:: shell

      $ omniperf profile --help

See :doc:`profile-mode` to learn about this mode in depth and to get started
profiling with Omniperf.

.. _modes-analyze:

Analyze mode
------------

``analyze``
   Loads profiling data from the ``--path`` (``-p``) directory into the Omniperf
   CLI analyzer where you have immediate access to profiling results and
   generated metrics. It generates metrics from the entirety of your profiled
   application or a subset identified through the Omniperf CLI analysis filters.

   To generate a lightweight GUI interface, you can add the `--gui` flag to your
   analysis command.

   This mode is a middle ground to the highly detailed Omniperf Grafana GUI and
   is great if you want immediate access to a hardware component you’re already
   familiar with.

   .. code-block:: shell

      $ omniperf analyze --help

See :doc:`analyze-mode` to learn about this mode in depth and to get started
with analysis using Omniperf.

.. _modes-database:

Database mode
-------------

``database``
   The :doc:`Grafana GUI dashboard <../install/grafana-setup>` is built on a
   MongoDB database. ``--import`` profiling results to the DB to interact with
   the workload in Grafana or `--remove` the workload from the DB.

   Connection options need to be specified. See :ref:`grafana-gui-import` for
   more details.

   .. code-block:: shell

      $ omniperf database --help

See :doc:`grafana-setup` to learn about setting up a Grafana server and database
instance to make your profiling data more digestible and shareable.

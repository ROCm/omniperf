.. _high-level-design:

High-level design
=================

The architecture of Omniperf consists of three major components shown in the
following diagram.

Omniperf profiling
   Acquires raw performance counters via application replay using ``rocprof``.
   Counters are stored in a comma-separated values format for further analysis.
   It runs a set of accelerator-specific micro benchmarks to acquire
   hierarchical roofline data. The roofline model is not available on
   accelerators pre-MI200.

Grafana analyzer for Omniperf
   * **Grafana database import**: All raw performance counters are imported into
     the a backend MongoDB database to support analysis and visualization in the
     Grafana GUI. Compatibility with previously generated data using older
     Omniperf versions is not guaranteed.
   * **Grafana dashboard GUI**: The Grafana dashboard retrieves the raw counters
     information from the backend database. It displays the relevant performance
     metrics and visualization.

Omniperf standalone GUI analyzer
   Omniperf provides a standalone GUI to enable basic performance analysis
   without the need to import data into a database instance.

.. figure:: ./data/omniperf_server_vs_client_install.png
   :align: center
   :alt: Architectural design of Omniperf


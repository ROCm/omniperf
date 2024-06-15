*******************
High-level overview
*******************

Omniperf is architecturally composed of three major components, as shown in the following figure.

.. list-table::
   :widths: 2, 3

   * - **Omniperf profiling**
     - Hello

   * - **Omniperf Grafana analyzer**
     - .. list-table::

          * - Grafana database import
            - Grafana GUI analyzer

          * - All raw performance
            - A Grafana dashboard

   * - **Omniperf standalone GUI analyzer**
     - A standalone GUI is provided to enable performance analysis without importing data into the backend database.

To learn more about Omniperf's client-server model, see Deployment.

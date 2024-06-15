****************
Install Omniperf
****************

Omniperf consists of two installation components:

* :ref:`Omniperf client-side <install-client-side>`

    * Provides core application profiling capability.
    * Allows collection of performance counters, filtering by hardware block,
      dispatch, kernel, and more.
    * Provides CLI-based analysis mode.
    * Provides standalone web interface for importing analysis metrics.

* :ref:`Omniperf server-side <install-server-side>`

    * Hosts the MongoDB backend and Grafana instance.
    * Packaged in a Docker container for easy setup.

Determine what you need to install based on how you would like to interact with
Omniperf. See the following decision tree to help determine what installation is
right for you.

.. _install-client-side:

Client-side installation
========================

Omniperf client-side requires the following basic software dependencies prior to usage:

* Python ``>= 3.8``
* CMake ``>= 3.19``
* ROCm ``>= 5.7.1``

In addition, Omniperf leverages a number of Python packages that are
documented in the top-level ``requirements.txt`` file. These must be
installed *before* configuring Omniperf.

.. admonition:: Optional packages for developers

   If you would like to build Omniperf as a developer, consider these additional
   requirements:

   .. list-table::
       :header-rows: 1

       * - Requirement file
         - Description

       * - requirements-doc.txt **CHANGE ME**
         - Python packages required to build docs from source.

       * - requirements-test.txt
         - Python packages required to run Omniperf's CI suite via PyTest.

   The recommended procedure for Omniperf usage is to install into a shared file
   system so that multiple users can access the final installation. The
   following steps illustrate how to install the necessary Python dependencies
   using `pip <https://packaging.python.org/en/latest/>`_ and Omniperf into a
   shared location controlled by the ``INSTALL_DIR`` environment variable.

Configuration variables
-----------------------
The following installation example leverages several
[CMake](https://cmake.org/cmake/help/latest/) project variables
defined as follows:

.. list-table::
    :header-rows: 1

    * - Variable name
      - Description

    * - ``CMAKE_INSTALL_PREFIX``
      - Controls the install path for Omniperf files.

    * - ``PYTHON_DEPS``
      - Specifies an optional path to resolve Python package dependencies.

    * - ``MOD_INSTALL_PATH``
      - Specifies an optional path for separate Omniperf modulefile installation.

A typical install will begin by downloading the latest release tarball
available from the
[Releases](https://github.com/ROCm/omniperf/releases) section
of the Omniperf development site. From there, untar and descend into
the top-level directory as follows:

```shell-session
$ tar xfz omniperf-v{__VERSION__}.tar.gz
$ cd omniperf-v{__VERSION__}
```

Next, install Python dependencies and complete the Omniperf configuration/install process as follows:

```shell-session
# define top-level install path
$ export INSTALL_DIR=<your-top-level-desired-install-path>

# install python deps
$ python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt

# configure Omniperf for shared install
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/{__VERSION__} \
        -DPYTHON_DEPS=${INSTALL_DIR}/python-libs \
        -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles ..

# install
$ make install
```

```{tip}
You may require `sudo` during the final install step if you
do not have write access to the chosen install path.
```


After completing these steps, a successful top-level installation directory looks as follows:
```shell-session
$ ls $INSTALL_DIR
modulefiles  {__VERSION__}  python-libs
```

### Execution using modulefiles

The installation process includes creation of an environment
modulefile for use with [Lmod](https://lmod.readthedocs.io). On
systems that support Lmod, a user can register the Omniperf modulefile
directory and setup their environment for execution of Omniperf as
follows:



```shell-session
$ module use $INSTALL_DIR/modulefiles
$ module load omniperf
$ which omniperf
/opt/apps/omniperf/{__VERSION__}/bin/omniperf

$ omniperf --version
ROC Profiler:   /opt/rocm-5.1.0/bin/rocprof

omniperf (v{__VERSION__})
```

```{tip} Users relying on an Lmod Python module locally may wish to
customize the resulting Omniperf modulefile post-installation to
include additional module dependencies.
```

### Execution without modulefiles

To use Omniperf without the companion modulefile, update your `PATH`
settings to enable access to the command-line binary. If you installed Python
dependencies in a shared location, update your `PYTHONPATH` config as well:

```shell-session
export PATH=$INSTALL_DIR/{__VERSION__}/bin:$PATH
export PYTHONPATH=$INSTALL_DIR/python-libs
```

### rocProf

Omniperf relies on a rocProf binary during the profiling
process. Normally the path to this binary will be detected
automatically, but it can also be overridden via the setting the
optional `ROCPROF` environment variable to the path of the binary the user
wishes to use instead.





%%% ### Generate Packaging
%%% ```console
%%% cd build
%%% cpack -G STGZ
%%% cpack -G DEB -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omniperf
%%% cpack -G RPM -D CPACK_PACKAGING_INSTALL_PREFIX=/opt/omniperf
%%% ```

.. _install-server-side:

Server-side installation
========================

Hello this abc def ghi jkl mnop qrst uvw xyz.

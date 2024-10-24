# Changelog for Omniperf

Full documentation for Omniperf is available at [https://rocm.docs.amd.com/projects/omniperf/en/latest/](https://rocm.docs.amd.com/projects/omniperf/en/latest/).

## Omniperf 2.0.1 for ROCm 6.2.0

### Changes

  * new option to force hardware target via `OMNIPERF_ARCH_OVERRIDE` global (#370)
  * CI/CD support for MI300 hardware (#373)
  * support for MI308X hardware (#375)

### Optimizations

  * cmake build improvements (#374)

## Omniperf 2.0.0 (17 May 2024)

  * improved logging than spans all modes (#177) (#317) (#335) (#341)
  * overhauled CI/CD that spans all modes (#179)
  * extensible SoC classes to better support adding new hardware configs (#180)
  * --kernel-verbose no longer overwrites kernel names (#193)
  * general cleanup and improved organization of source code (#200) (#210) 
  * separate requirement files for docs and testing dependencies (#205) (#262) (#358)
  * add support for MI300 hardware (#231)
  * upgrade Grafana assets and build script to latest release (#235)
  * update minimum ROCm and Python requirements (#277)
  * sort rocprofiler input files prior to profiling (#304)
  * new --quiet option will suppress verbose output and show a progress bar (#308)
  * roofline support for Ubuntu 22.04 (#319)

## Omniperf 1.1.0-PR1 (13 Oct 2023)

  * standardize headers to use 'avg' instead of 'mean'
  * add color code thresholds to standalone gui to match grafana
  * modify kernel name shortener to use cpp_filt (#168)
  * enable stochastic kernel dispatch selection (#183)
  * patch grafana plugin module to address a known issue in the latest version (#186)
  * enhanced communication between analyze mode kernel flags (#187)

## Omniperf 1.0.10 (22 Aug 2023)

  * critical patch for detection of llvm in rocm installs on SLURM systems

## Omniperf 1.0.9 (17 Aug 2023)

  * add units to L2 per-channel panel (#133)
  * new quickstart guide for Grafana setup in docs (#135)
  * more detail on kernel and dispatch filtering in docs (#136, #137)
  * patch manual join utility for ROCm >5.2.x (#139)
  * add % of peak values to low level speed-of-light panels (#140)
  * patch critical bug in Grafana by removing a deprecated plugin (#141)
  * enhancements to KernelName demangeler (#142)
  * general metric updates and enhancements (#144, #155, #159)
  * add min/max/avg breakdown to instruction mix panel (#154)

## Omniperf 1.0.8 (30 May 2023)

  * add `--kernel-names` option to toggle kernelName overlay in standalone roofline plot (#93)
  * remove unused python modules (#96)
  * fix empirical roofline calculation for single dispatch workloads (#97)
  * match color of arithmetic intensity points to corresponding bw lines

  * ux improvements in standalone GUI (#101)
  * enhanced readability for filtering dropdowns in standalone GUI (#102)
  * new logfile to capture rocprofiler output (#106)
  * roofline support for sles15 sp4 and future service packs (#109)
  * adding dockerfiles for all supported Linux distros
  * new examples for `--roof-only` and `--kernel` options added to documentation
  
  * enable cli analysis in Windows (#110)
  * optional random port number in standalone GUI (#111)
  * limit length of visible kernelName in `--kernel-names` option (#115)
  * adjust metric definitions (#117, #130)
  * manually merge rocprof runs, overriding default rocprofiler implementation (#125)
  * fixed compatibility issues with Python 3.11 (#131)
  
## Omniperf 1.0.8-PR2 (17 Apr 2023)

  * ux improvements in standalone GUI (#101)
  * enhanced readability for filtering dropdowns in standalone GUI (#102)
  * new logfile to capture rocprofiler output (#106)
  * roofline support for sles15 sp4 and future service packs (#109)
  * adding dockerfiles for all supported Linux distros
  * new examples for `--roof-only` and `--kernel` options added to documentation

## Omniperf 1.0.8-PR1 (13 Mar 2023)

  * add `--kernel-names` option to toggle kernelName overlay in standalone roofline plot (#93)
  * remove unused python modules (#96)
  * fix empirical roofline calculation for single dispatch workloads (#97)
  * match color of arithmetic intensity points to corresponding bw lines
  
## Omniperf 1.0.7 (21 Feb 2023)

  * update documentation (#52, #64)
  * improved detection of invalid command line arguments (#58, #76)
  * enhancements to standalone roofline (#61)
  * enable Omniperf on systems with X-server (#62)
  * raise minimum version requirement for rocm (#64)
  * enable baseline comparison in CLI analysis (#65)
  * add multi-normalization to new metrics (#68, #81)
  * support alternative profilers (#70)
  * add MI100 configs to override rocprofiler's incomplete default (#75)
  * improve error message when no GPU(s) detected (#85)
  * separate CI tests by Linux distro and add status badges
  
## Omniperf 1.0.6 (21 Dec 2022)

  * CI update: documentation now published via github action (#22)
  * better error detection for incomplete ROCm installs (#56)

## Omniperf 1.0.5 (13 Dec 2022)

  * store application command-line parameters in profiling output (#27)
  * enable additional normalizations in CLI mode (#30)
  * add missing ubuntu 20.04 roofline binary to packaging (#34)
  * update L1 bandwidth metric calculations (#36)
  * add L1 <-> L2 bandwidth calculation (#37)
  * documentation updates (#38, #41)
  * enhanced subprocess logging to identify critical errors in rocprofiler (#50)
  * maintain git sha in production installs from tarball (#53)

## Omniperf 1.0.4 (11 Nov 2022)

  * update python requirements.txt with minimum versions for numpy and pandas
  * addition of progress bar indicator in web-based GUI (#8)
  * reduced default content for web-based GUI to reduce load times (#9)
  * minor packaging and CI updates 
  * variety of documentation updates  
  * added an optional argument to vcopy.cpp workload example to specify device id

## Omniperf 1.0.3 (07 Nov 2022)

  * initial Omniperf release

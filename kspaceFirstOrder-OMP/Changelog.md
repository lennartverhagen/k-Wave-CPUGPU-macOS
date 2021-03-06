## Version 2.17 (28 February 2020) - k-Wave release 1.3
  - Added support for 2D medium.
  - Added support for axisymmetric medium.
  - Added support for Stokes absorption for normal medium.
  - Added new source scaling term (additive) and the old one renamed to
    additive-no-correction.
  - Added more verbosity reporting source flags and expected memory consumption
    as well as sensor flags and expected file consumption.
  - Added copy sensor mask to the output file.
  - Added checkpoint interval in time steps.
  - Added import of system-wide wisdom for FFTW.
  - Added setting of default number of threads according to OMP_NUM_THREADS
    variable, if defined.
  - Changed input file version to 1.2 (many quantities are now generated by the
    code, e.g., PML, derivative operators, shift variables, etc).
  - Changed default alignment to 64b.
  - Fixed compatibility issues with Visual Studio 2017 and Intel 2019.
  - Removed generated doxygen documentation from the repository.

## Version 2.16 (04 September 2017) - k-Wave release 1.2
  - Introduction of SC@FIT coding style.
  - Added explicit calculation of nonlinear term.
  - Added new visual style and verbose levels.
  - Added better support for C++-11.
  - Added OpenMP 4.0 SIMD vectorization of kernels.
  - Added Parallel source injection.
  - Bugfix reading scalar flags.
  - Bugfix with transducer delay mask after checkpoint.
  - Bugfix with percentage report.

## Version 2.15 (02 October 2013) - k-Wave release 1.1
  - Removed -I output.
  - Added -u_non_staggered_raw.
  - Merged source bases for Windows, Linux, Intel and GNU compilers.
  - Updated doxyen documentation.
  - Added use of standard math library.
  - Added git hash and other useful information to the --version report.
  - Added wisdom import for checkpoint-restart.
  - Changed index matrices data types to size_t (used to be long).
  - Changed default compression level is to 0.
  - Added a header to the checkpoint file.
  - Fixed aggregated values initialization.

## Version 2.14 (21 January 2013) - k-Wave release 1.0
  - Converted from Mercurial to Git.
  - Bugfix with -r verbose interval.
  - Added flag --p_min, --u_min, --p_max_all, --p_min_all, --u_max_all,
    --u_min_all.
  - Fixed missing <unistd.h> for gcc/4.7.
  - Changed use _OPENMP compiler flag for OpenMP instead of __NO_OMP__.
  - Added cuboid corners sensor mask.
  - Changed output quantities to be managed by OutputStream.
  - Added check for HDF5 file version.
  - Added proper alignment for SSE and AVX.
  - Added checkpoint-restart functionality.
    
## Version 2.13 (19 September 2012)
  - Start index begins from 1.
  - Changed p_avg and u_avg to p_rms, u_rms.
  - Added --u_final (p_final, and u_final stores entire matrices).
  - Intensity calculated base on the temporal and spatial staggered grid.
  - Changed log from FFTW to FFT.
  - Moved constants and typedefs  into classes.
  - Added doxygen documentation.

## Version 2.12 (09 August 2012)
  - HDF5 input files introduced (HDF Sequential 1.8.9).
  - HDF output files introduced (all outputs in a single file).
  - Added parameter -c (compression level).
  - Matlab is not needed any more.
  - Changed terminal output format.
  - Removed transducer source duplicity bug.
  - Variables renaming.
  - Removed size scalars.
  - Kappa, absorb_nabla1, absorb_nabla2,absorb_eta,absorb_tau generated at
    the beginning.
  - Added scalars.
  - Added linear and lossless case.
  - New commandline parameters.
  - Error messages in a single file.
  - All matrix names in a single file.
  - Created mercurial repository.

## Version 2.11 (11 July 2012)
  - The framework has been rewritten.
  - Added common ancestor for all matrices and the Matrix container.
  - Dimension modification (stored logically according to Matlab).
  - All obsolete parts and unnecessary parts of the code were removed.

## Version 2.10 (19 June 2012)
  - Added non-uniform grid (in case of ux_sgx added into dt./rho0_sgx).

## Version 2.9 (16 February 2012)
  - Added new u source conditions.
  - Added new p source conditions.
  - Some matrices renamed.
  - New parameters.
  - Saving of u matrices.
  - Saving of max pressure.
  - Adding of parameter -f to store only last iteration of p.
  - BonA is now a 3D matrix.
  - Changed timing.
  - Fixed bug in constructor.

## Version 2.7 (8 December 2011)
  - Direct import from Matlab data files.
  - Export to a binary file.
  - Modified TParameters and Command line parameters.

## Version 2.4 (28 September 2011)
  - The code was tidied up.
  - Add c square.
  - Affinity, better way.

## Version 2.3 (22 September 2011)
  - Affinity, first touch strategy.
  - Pragma for fusion.
  - Added three temp real matrices (no memory allocation during execution any
    more).
  - Removed PAPI interface.
  - Removed TAU interface.
  

## Version 2.2 (5 September 2011) - debug version
  - Included PAPI interface.
  - Included TAU interface.
  - disabled C2C fftw plans.
  - fixed progress calculation bug when running with fewer then 100 time steps.
  - added flush to progress.

## Version 2.1 (23 August 2011)
  - Complex2Complex fft_3D (ifft_3D) replaced with Real2Complex and Complex2Real
    ones.
  - All complex computation reduced to number of elements of Nx_r.
  - Added Nx_r, dimension sizes.
  - Altered progress output format.
  - added parameter -t (thead count) -p (print interval) -c
   (configuration_file).

## Version 2.0 (16 August 2011)
  - Added new code to handle transducer_input_signal, us_index, delay_mask,
    transducer_source.
  - Added TOutputStreamMatrix maintaining results and store them on HDD in a
    stream fashion.
  - Modified configuration.xml and the parameters organization.
  - Default number of threads set to be the number of physical cores.

## Version 1.1 (12 August 2011)
  - Operation fusion - more tightly closed operations placed under 1 loop cycle.
  - Added TRealMatrixData::CopyFromComplexSameSizeAndNormalize (7% improvement).
  - fftwf_plan_with_nthreads() revised.
  - fftwf_init_threads() bug removed.

## Version 1.0 (10 August 2011)
  - basic implementation of KSpaceFirstOrder3D - nonlinear.
  - based on KSpace3DBareBone 2.1.
  

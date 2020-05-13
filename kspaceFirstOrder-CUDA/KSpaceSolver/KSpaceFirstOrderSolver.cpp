/**
 * @file      KSpaceFirstOrderSolver.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the main class of the project responsible for the entire simulation.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 July      2012, 10:27 (created)\n
 *            11 February  2020, 16:14 (revised)
 *
 * @copyright Copyright (C) 2012 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

// Linux build
#ifdef __linux__
  #include <sys/resource.h>
#endif

// Windows build
#ifdef _WIN64
  #define _USE_MATH_DEFINES
  #include <Windows.h>
  #include <Psapi.h>
  #pragma comment(lib, "Psapi.lib")
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <cmath>
#include <limits>

#include <KSpaceSolver/KSpaceFirstOrderSolver.h>
#include <KSpaceSolver/SolverCudaKernels.cuh>
#include <Containers/MatrixContainer.h>
#include <Logger/Logger.h>

using std::ios;
/// Shortcut for Simulation dimensions.
using SD = Parameters::SimulationDimension;
/// Shortcut for Matrix id in the container.
using MI = MatrixContainer::MatrixIdx;
/// Shortcut for Output stream id in the container.
using OI = OutputStreamContainer::OutputStreamIdx;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Initialization of the static map with header attribute names.
 * If c0 is heterogeneous, absorption must be heterogeneous as well (3rd cases).
 * Not all cases are used at this moment.
 */
KSpaceFirstOrderSolver::ComputeMainLoopImp KSpaceFirstOrderSolver::sComputeMainLoopImp
{
  //  3D cases          SD  rho0   bOnA   c0     alpha
  {std::make_tuple(SD::k3D, true,  true,  true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  true,  true , true >},
  {std::make_tuple(SD::k3D, true,  true,  true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  true,  true , false>},
  {std::make_tuple(SD::k3D, true,  true,  false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  true,  false, false>},
  {std::make_tuple(SD::k3D, true,  true,  false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  true,  false, false>},

  {std::make_tuple(SD::k3D, true,  false, true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  false, true , true >},
  {std::make_tuple(SD::k3D, true,  false, true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  false, true , false>},
  {std::make_tuple(SD::k3D, true,  false, false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  false, false, false>},
  {std::make_tuple(SD::k3D, true,  false, false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, true,  false, false, false>},

  {std::make_tuple(SD::k3D, false, true,  true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, true,  true,  true >},
  {std::make_tuple(SD::k3D, false, true,  true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, true,  true,  false>},
  {std::make_tuple(SD::k3D, false, true,  false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, true,  false, false>},
  {std::make_tuple(SD::k3D, false, true,  false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, true,  false, false>},

  {std::make_tuple(SD::k3D, false, false, true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, false, true , true >},
  {std::make_tuple(SD::k3D, false, false, true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, false, true , false>},
  {std::make_tuple(SD::k3D, false, false, false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, false, false, false>},
  {std::make_tuple(SD::k3D, false, false, false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k3D, false, false, false, false>},


  //  2D cases          SD  rho0   bOnA   c0     alpha
  {std::make_tuple(SD::k2D, true,  true,  true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  true,  true,  true >},
  {std::make_tuple(SD::k2D, true,  true,  true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  true,  true,  false>},
  {std::make_tuple(SD::k2D, true,  true,  false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  true,  false, false>},
  {std::make_tuple(SD::k2D, true,  true,  false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  true,  false, false>},

  {std::make_tuple(SD::k2D, true,  false, true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  false, true,  true >},
  {std::make_tuple(SD::k2D, true,  false, true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  false, true,  false>},
  {std::make_tuple(SD::k2D, true,  false, false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  false, false, false>},
  {std::make_tuple(SD::k2D, true,  false, false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, true,  false, false, false>},

  {std::make_tuple(SD::k2D, false, true,  true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, true,  true,  true >},
  {std::make_tuple(SD::k2D, false, true,  true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, true,  true,  false>},
  {std::make_tuple(SD::k2D, false, true,  false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, true,  false, false>},
  {std::make_tuple(SD::k2D, false, true,  false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, true,  false, false>},

  {std::make_tuple(SD::k2D, false, false, true,  true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, false, true,  true >},
  {std::make_tuple(SD::k2D, false, false, true,  false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, false, true,  false>},
  {std::make_tuple(SD::k2D, false, false, false, true ),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, false, false, false>},
  {std::make_tuple(SD::k2D, false, false, false, false),
                   &KSpaceFirstOrderSolver::computeMainLoop<SD::k2D, false, false, false, false>}
};

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor of the class.
 */
KSpaceFirstOrderSolver::KSpaceFirstOrderSolver()
  : mMatrixContainer(),
    mOutputStreamContainer(),
    mParameters(Parameters::getInstance()),
    mActPercent(0l),
    mIsTimestepRightAfterRestore(false),
    mTotalTime(), mPreProcessingTime(), mDataLoadTime(), mSimulationTime(), mPostProcessingTime(), mIterationTime()
{
  mTotalTime.start();

  // Switch off default HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);
}// end of KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor of the class.
 */
KSpaceFirstOrderSolver::~KSpaceFirstOrderSolver()
{
  // Delete CUDA FFT plans and related data
  CufftComplexMatrix::destroyAllPlansAndStaticData();

  // Free memory
  freeMemory();

  // Reset device after the run - recommended by CUDA SDK
  cudaDeviceReset();
}// end of ~KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

/**
 * The method allocates the matrix container, creates all matrices and creates all output streams.
 * The memory for output stream is not allocated since the size of the masks is not known.
 */
void KSpaceFirstOrderSolver::allocateMemory()
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtMemoryAllocation);
  Logger::flush(Logger::LogLevel::kBasic);

  // Add matrices into the container and create all matrices
  mMatrixContainer.init();
  mMatrixContainer.createMatrices();

  // Add output streams into container
  mOutputStreamContainer.init(mMatrixContainer);

  Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * The method frees all memory allocated by the class.
 */
void KSpaceFirstOrderSolver::freeMemory()
{
  mMatrixContainer.freeMatrices();
  mOutputStreamContainer.freeStreams();
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Load data from the input file provided by the Parameter class and create the output time series streams.
 * If checkpointing is enabled, this may include reading data from checkpoint and output files.
 */
void KSpaceFirstOrderSolver::loadInputData()
{
  // Load data from disk
  Logger::log(Logger::LogLevel::kBasic, kOutFmtDataLoading);
  Logger::flush(Logger::LogLevel::kBasic);
  // Start timer
  mDataLoadTime.start();

  // Get handles
  Hdf5File& inputFile      = mParameters.getInputFile(); // file is opened (in Parameters)
  Hdf5File& outputFile     = mParameters.getOutputFile();
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();

  // Load data from disk
  Logger::log(Logger::LogLevel::kFull, kOutFmtNoDone);
  Logger::log(Logger::LogLevel::kFull, kOutFmtReadingInputFile);
  Logger::flush(Logger::LogLevel::kFull);

  // Load data from the input file
  mMatrixContainer.loadDataFromInputFile();

  // Close the input file since we don't need it anymore.
  inputFile.close();

  Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

  // Recover from the checkpoint if it is enabled and the checkpoint file exists.
  bool recoverFromCheckpoint = (mParameters.isCheckpointEnabled() &&
                                Hdf5File::canAccess(mParameters.getCheckpointFileName()));

  if (recoverFromCheckpoint)
  {
    //--------------------------------------- Recovery from checkpoint file ------------------------------------------//
    Logger::log(Logger::LogLevel::kFull, kOutFmtReadingCheckpointFile);
    Logger::flush(Logger::LogLevel::kFull);

    // Open checkpoint file
    checkpointFile.open(mParameters.getCheckpointFileName());
    // Check the checkpoint file
    checkCheckpointFile();

    // Read the actual value of t_index
    size_t checkpointedTimeIndex;
    checkpointFile.readScalarValue(checkpointFile.getRootGroup(),
                                   mParameters.getTimeIndexHdf5Name(),
                                   checkpointedTimeIndex);
    mParameters.setTimeIndex(checkpointedTimeIndex);

    // Read necessary matrices from the checkpoint file
    mMatrixContainer.loadDataFromCheckpointFile();
    // Close checkpoint file, not needed any more.
    checkpointFile.close();

    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);


    // Read data from the output file
    Logger::log(Logger::LogLevel::kFull, kOutFmtReadingOutputFile);
    Logger::flush(Logger::LogLevel::kFull);

    // Reopen output file for RW access
    outputFile.open(mParameters.getOutputFileName(), H5F_ACC_RDWR);
    // Read file header of the output file
    mParameters.getFileHeader().readHeaderFromOutputFile(outputFile);
    // Check the output file
    checkOutputFile();
    // Restore elapsed time
    readElapsedTimeFromOutputFile();
    // Reopen steams to store data again.
    mOutputStreamContainer.reopenStreams();

    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
  }
  else
  { //------------------------------------ First round of multi-leg simulation ---------------------------------------//
    // Create the output file
    Logger::log(Logger::LogLevel::kFull, kOutFmtCreatingOutputFile);
    Logger::flush(Logger::LogLevel::kFull);

    outputFile.create(mParameters.getOutputFileName());
    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

    // Create the steams, link them with the sampled matrices, however DO NOT allocate memory!
    mOutputStreamContainer.createStreams();
  }

  // Stop timer
  mDataLoadTime.stop();
  if (Logger::getLevel() != Logger::LogLevel::kFull)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }

  // Print out loading time
  Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mDataLoadTime.getElapsedTime());
}// end of loadInputData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Run the simulation.
 */
void KSpaceFirstOrderSolver::compute()
{
  mPreProcessingTime.start();

  Logger::log(Logger::LogLevel::kBasic, kOutFmtFftPlans);
  Logger::flush(Logger::LogLevel::kBasic);

  CudaParameters& cudaParameters = mParameters.getCudaParameters();

  // FFT initialization and preprocessing
  try
  {
    // Initialize all cuda FFT plans
    initializeCufftPlans();
    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);

    Logger::log(Logger::LogLevel::kBasic,kOutFmtPreProcessing);
    Logger::flush(Logger::LogLevel::kBasic);

    // Preprocessing phase generating necessary variables
    if (mParameters.isSimulation3D())
    {
      preProcessing<SD::k3D>();
    }
    else
    {
      preProcessing<SD::k2D>();
    }

    mPreProcessingTime.stop();
    // Set kernel configurations
    cudaParameters.setKernelConfiguration();

    // Set up constant memory - copy over to GPU
    // Constant memory uses some variables calculated during preprocessing
    cudaParameters.setUpDeviceConstants();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
  }
  catch (const std::exception& e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(),kErrFmtPathDelimiters, 9));
  }

  // Logger header for simulation
  Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPreProcessingTime.getElapsedTime());

  // Memory and disk space
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCompResourcesHeader);
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentHostMemory,   getHostMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtCurrentDeviceMemory, getDeviceMemoryUsage());
  Logger::log(Logger::LogLevel::kBasic, kOutFmtOutputFileUsage,     getFileUsage());

  // CUDA block and thread sizes
  const std::string blockDims = Logger::formatMessage(kOutFmtCudaGridShapeFormat,
                                                      cudaParameters.getSolverGridSize1D(),
                                                      cudaParameters.getSolverBlockSize1D());

  Logger::log(Logger::LogLevel::kFull, kOutFmtCudaSolverGridShape, blockDims.c_str());

  const std::string gridDims = Logger::formatMessage(kOutFmtCudaGridShapeFormat,
                                                     cudaParameters.getSamplerGridSize1D(),
                                                     cudaParameters.getSamplerBlockSize1D());

  Logger::log(Logger::LogLevel::kFull, kOutFmtCudaSamplerGridShape, gridDims.c_str());

  // Main simulation loop
  try
  {
    mSimulationTime.start();

    // Invoke a specific version of computeMainLoop
    sComputeMainLoopImp[std::make_tuple(mParameters.getSimulationDimension(),
                                        mParameters.getRho0ScalarFlag(),
                                        mParameters.getBOnAScalarFlag(),
                                        mParameters.getC0ScalarFlag(),
                                        mParameters.getAlphaCoeffScalarFlag())](*this);

    mSimulationTime.stop();

    Logger::log(Logger::LogLevel::kBasic,kOutFmtSimulationEndSeparator);
  }
  catch (const std::exception& e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSimulatoinFinalSeparator);
    Logger::errorAndTerminate(Logger::wordWrapString(e.what(),kErrFmtPathDelimiters, 9));
  }

  // Post processing phase
  mPostProcessingTime.start();

  try
  {
    if (isCheckpointInterruption())
    { // Checkpoint
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointCompletedTimeSteps, mParameters.getTimeIndex());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCheckpointHeader);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtCreatingCheckpoint);
      Logger::flush(Logger::LogLevel::kBasic);

      if (Logger::getLevel() == Logger::LogLevel::kFull)
      {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtNoDone);
      }

      writeCheckpointData();

      if (Logger::getLevel() != Logger::LogLevel::kFull)
      {
        Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
      }
    }
    else
    { // Finish
      Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mSimulationTime.getElapsedTime());
      Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);
      Logger::log(Logger::LogLevel::kBasic, kOutFmtPostProcessing);
      Logger::flush(Logger::LogLevel::kBasic);

      postProcessing();

      // If checkpointing is enabled and the checkpoint file already exists, delete it
      if (mParameters.isCheckpointEnabled())
      {
        std::remove(mParameters.getCheckpointFileName().c_str());
      }
      Logger::log(Logger::LogLevel::kBasic, kOutFmtDone);
    }
  }
  catch (const std::exception &e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtFailed);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);

    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters,9));
  }
  mPostProcessingTime.stop();

  // Final data written
  try
  {
    writeOutputDataInfo();
    mParameters.getOutputFile().close();

    Logger::log(Logger::LogLevel::kBasic, kOutFmtElapsedTime, mPostProcessingTime.getElapsedTime());
    }
  catch (const std::exception &e)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLastSeparator);
    Logger::errorAndTerminate(Logger::wordWrapString(e.what(), kErrFmtPathDelimiters, 9));
  }
}// end of compute
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak host memory usage.
 */
size_t KSpaceFirstOrderSolver::getHostMemoryUsage() const
{
  // Linux build
  #ifdef __linux__
    struct rusage memUsage;
    getrusage(RUSAGE_SELF, &memUsage);

    return memUsage.ru_maxrss >> 10;
  #endif

  // Windows build
  #ifdef _WIN64
    HANDLE hProcess;
    PROCESS_MEMORY_COUNTERS pmc;

    hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                           FALSE,
                           GetCurrentProcessId());

    GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc));
    CloseHandle(hProcess);

    return pmc.PeakWorkingSetSize >> 20;
  #endif
}// end of getHostMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get peak device memory usage.
 */
size_t KSpaceFirstOrderSolver::getDeviceMemoryUsage() const
{
  size_t free, total;
  cudaMemGetInfo(&free, &total);

  return ((total - free) >> 20);
}// end of getDeviceMemoryUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get expected output file usage in MB.
 */
size_t KSpaceFirstOrderSolver::getFileUsage()
{
  // Accumulator
  size_t fileUsage = 0;

  // Sensor sizes and time steps sampled
  const size_t sensorMaskSize = (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
                                   ? mParameters.getSensorMaskIndexSize()
                                   : getIndexMatrix(MI::kSensorMaskCorners).getSizeOfAllCuboids();
  const size_t domainSize     = mParameters.getFullDimensionSizes().nElements();
  const size_t sampledSteps   = mParameters.getNt() - mParameters.getSamplingStartTimeIndex();

  // Time series
  if (mParameters.getStorePressureRawFlag())
  {
    fileUsage += sensorMaskSize * sampledSteps;
  }

  if (mParameters.getStoreVelocityRawFlag())
  {
    fileUsage += 3 * sensorMaskSize * sampledSteps;
  }

  if (mParameters.getStoreVelocityNonStaggeredRawFlag())
  {
    fileUsage += 3 * sensorMaskSize * sampledSteps;
  }

  // Aggregated pressure
  if (mParameters.getStorePressureRmsFlag())
  {
    fileUsage += sensorMaskSize;
  }
  if (mParameters.getStorePressureMaxFlag())
  {
    fileUsage += sensorMaskSize;
  }
  if (mParameters.getStorePressureMinFlag())
  {
    fileUsage += sensorMaskSize;
  }

  if (mParameters.getStorePressureMaxAllFlag())
  {
    fileUsage += domainSize;
  }
  if (mParameters.getStorePressureMinAllFlag())
  {
    fileUsage += domainSize;
  }
  if (mParameters.getStorePressureFinalAllFlag())
  {
    fileUsage += domainSize;
  }

  // Aggregated velocities
  if (mParameters.getStoreVelocityRmsFlag())
  {
    fileUsage += 3 * sensorMaskSize;
  }
  if (mParameters.getStoreVelocityMaxFlag())
  {
    fileUsage += 3 * sensorMaskSize;
  }
  if (mParameters.getStoreVelocityMinFlag())
  {
    fileUsage += 3 * sensorMaskSize;
  }

  if (mParameters.getStoreVelocityMaxAllFlag())
  {
    fileUsage += 3 * domainSize;
  }
  if (mParameters.getStoreVelocityMinAllFlag())
  {
    fileUsage += 3 * domainSize;
  }
  if (mParameters.getStoreVelocityFinalAllFlag())
  {
    fileUsage += 3 * domainSize;
  }

  // Sensor mask
  if (mParameters.getCopySensorMaskFlag())
  {
    // 2x due to the size_t data type
    fileUsage = (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
                   ? fileUsage + 2 * sensorMaskSize
                   : fileUsage + 2 * mParameters.getSensorMaskCornersSize() * 6;
  }

  return float(fileUsage) / float(1024 * 1024 / 4);

}// end of getFileUsage
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get release code version.
 */
std::string KSpaceFirstOrderSolver::getCodeName() const
{
  return kOutFmtKWaveVersion;
}// end of getCodeName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print full code name and the license.
 */
void KSpaceFirstOrderSolver::printFullCodeNameAndLicense() const
{
  Logger::log(Logger::LogLevel::kBasic, kOutFmtBuildNoDataTime, 10, 11, __DATE__, 8, 8, __TIME__);

  if (mParameters.getGitHash() != "")
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtVersionGitHash, mParameters.getGitHash().c_str());
  }
  Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);

  // OS detection
  #ifdef __linux__
    Logger::log(Logger::LogLevel::kBasic, kOutFmtLinuxBuild);
  #elif __APPLE__
    Logger::log(Logger::LogLevel::kBasic, kOutFmtMacOsBuild);
  #elif _WIN32
    Logger::log(Logger::LogLevel::kBasic, kOutFmtWindowsBuild);
  #endif

  // Compiler detections
  #if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtGnuCompiler, __VERSION__);
  #endif
  #ifdef __INTEL_COMPILER
    Logger::log(Logger::LogLevel::kBasic, kOutFmtIntelCompiler, __INTEL_COMPILER);
  #endif
  #ifdef _MSC_VER
	Logger::log(Logger::LogLevel::kBasic, kOutFmtVisualStudioCompiler, _MSC_VER);
  #endif

  // CPU detection
  Logger::log(Logger::LogLevel::kBasic, kOutFmtProcessorNameLeft, mParameters.getProcessorName().c_str());

  // Default branch instruction set detection.
  #if (defined (__AVX512F__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX512);
  #elif (defined (__AVX2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX2);
  #elif (defined (__AVX__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtAVX);
  #elif (defined (__SSE4_2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE42);
  #elif (defined (__SSE4_1__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE41);
  #elif (defined (__SSE3__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE3);
  #elif (defined (__SSE2__))
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSSE2);
  #endif

  Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);

  // CUDA detection
  int cudaRuntimeVersion;
  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaRuntimeNA);
  }
  else
  {
    Logger::log(Logger::LogLevel::kBasic,
                ((cudaRuntimeVersion / 1000) < 10) ? kOutFmtCudaRuntime : kOutFmtCudaRuntime10,
                cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10);
  }

  int cudaDriverVersion;
  cudaDriverGetVersion(&cudaDriverVersion);
  Logger::log(Logger::LogLevel::kBasic,
              ((cudaDriverVersion / 1000) < 10) ? kOutFmtCudaDriver : kOutFmtCudaDriver10,
              cudaDriverVersion / 1000, (cudaDriverVersion % 100) / 10);

  const CudaParameters& cudaParameters = mParameters.getCudaParameters();
  // No GPU was found
  if (cudaParameters.getDeviceIdx() == CudaParameters::kDefaultDeviceIdx)
  {
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaDeviceInfoNA);
  }
  else
  {
    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaCodeArch,
                SolverCudaKernels<>::getCudaCodeVersion() / 10.f);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtSeparator);
    Logger::log(Logger::LogLevel::kBasic, kOutFmtCudaDevice, cudaParameters.getDeviceIdx());

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaDeviceName,
                cudaParameters.getDeviceName().c_str());

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtCudaCapability,
                cudaParameters.getDeviceProperties().major,
                cudaParameters.getDeviceProperties().minor);
  }

  // Print license
  Logger::log(Logger::LogLevel::kBasic, kOutFmtLicense);
}// end of printFullCodeNameAndLicense
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Check the output file has the correct format and version.
 */
void KSpaceFirstOrderSolver::checkOutputFile()
{
  // The header has already been read
  Hdf5FileHeader& fileHeader = mParameters.getFileHeader();
  Hdf5File&       outputFile = mParameters.getOutputFile();

  // Check file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kOutput)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadOutputFileFormat, mParameters.getOutputFileName().c_str()));
  }

  // Check file version
  if (!fileHeader.checkFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadFileVersion,
                                             fileHeader.getFileVersionName().c_str(),
                                             mParameters.getOutputFileName().c_str(),
                                             fileHeader.getCurrentFileVersionName().c_str()));
  }

  // Check dimension sizes
  DimensionSizes outputDimSizes;
  outputFile.readScalarValue(outputFile.getRootGroup(), mParameters.getNxHdf5Name(), outputDimSizes.nx);
  outputFile.readScalarValue(outputFile.getRootGroup(), mParameters.getNyHdf5Name(), outputDimSizes.ny);
  outputFile.readScalarValue(outputFile.getRootGroup(), mParameters.getNzHdf5Name(), outputDimSizes.nz);

  if (mParameters.getFullDimensionSizes() != outputDimSizes)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtOutputDimensionsMismatch,
                                             outputDimSizes.nx,
                                             outputDimSizes.ny,
                                             outputDimSizes.nz,
                                             mParameters.getFullDimensionSizes().nx,
                                             mParameters.getFullDimensionSizes().ny,
                                             mParameters.getFullDimensionSizes().nz));
  }
}// end of checkOutputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check the checkpoint file has the correct format and version.
 */
void KSpaceFirstOrderSolver::checkCheckpointFile()
{
  // Read the header and check the file version
  Hdf5FileHeader fileHeader;
  Hdf5File&      checkpointFile = mParameters.getCheckpointFile();

  fileHeader.readHeaderFromCheckpointFile(checkpointFile);

  // Check file type
  if (fileHeader.getFileType() != Hdf5FileHeader::FileType::kCheckpoint)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadCheckpointFileFormat,
                                             mParameters.getCheckpointFileName().c_str()));
  }

  // Check file version
  if (!fileHeader.checkFileVersion())
  {
    throw ios::failure(Logger::formatMessage(kErrFmtBadFileVersion,
                                             fileHeader.getFileVersionName().c_str(),
                                             mParameters.getCheckpointFileName().c_str(),
                                             fileHeader.getCurrentFileVersionName().c_str()));
  }

  // Check dimension sizes
  DimensionSizes checkpointDimSizes;
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), mParameters.getNxHdf5Name(), checkpointDimSizes.nx);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), mParameters.getNyHdf5Name(), checkpointDimSizes.ny);
  checkpointFile.readScalarValue(checkpointFile.getRootGroup(), mParameters.getNzHdf5Name(), checkpointDimSizes.nz);

  if (mParameters.getFullDimensionSizes() != checkpointDimSizes)
  {
    throw ios::failure(Logger::formatMessage(kErrFmtCheckpointDimensionsMismatch,
                                             checkpointDimSizes.nx,
                                             checkpointDimSizes.ny,
                                             checkpointDimSizes.nz,
                                             mParameters.getFullDimensionSizes().nx,
                                             mParameters.getFullDimensionSizes().ny,
                                             mParameters.getFullDimensionSizes().nz));
  }
}// end of checkCheckpointFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Restore cumulated elapsed time from the output file.
 */
void KSpaceFirstOrderSolver::readElapsedTimeFromOutputFile()
{
  double totalTime, dataLoadTime, preProcessingTime, simulationTime, postProcessingTime;

  // Get execution times stored in the output file header
  mParameters.getFileHeader().getExecutionTimes(totalTime,
                                                dataLoadTime,
                                                preProcessingTime,
                                                simulationTime,
                                                postProcessingTime);

  mTotalTime.SetElapsedTimeOverPreviousLegs(totalTime);
  mDataLoadTime.SetElapsedTimeOverPreviousLegs(dataLoadTime);
  mPreProcessingTime.SetElapsedTimeOverPreviousLegs(preProcessingTime);
  mSimulationTime.SetElapsedTimeOverPreviousLegs(simulationTime);
  mPostProcessingTime.SetElapsedTimeOverPreviousLegs(postProcessingTime);

}// end of readElapsedTimeFromOutputFile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Was the loop interrupted to checkpoint?
 */
bool KSpaceFirstOrderSolver::isCheckpointInterruption() const
{
  return (mParameters.getTimeIndex() != mParameters.getNt());
}// end of isCheckpointInterruption
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize cuda FFT plans.
 */
void KSpaceFirstOrderSolver::initializeCufftPlans()
{
  // Create real to complex plans
  CufftComplexMatrix::createR2CFftPlanND(mParameters.getFullDimensionSizes());

  // create complex to real plans
  CufftComplexMatrix::createC2RFftPlanND(mParameters.getFullDimensionSizes());

  // If necessary, create 1D shift plans.
  // In this case, the matrix has a bit bigger dimensions to be able to store shifted matrices.
  if (Parameters::getInstance().getStoreVelocityNonStaggeredRawFlag())
  {
    // X shifts
    CufftComplexMatrix::createR2CFftPlan1DX(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DX(mParameters.getFullDimensionSizes());

    // Y shifts
    CufftComplexMatrix::createR2CFftPlan1DY(mParameters.getFullDimensionSizes());
    CufftComplexMatrix::createC2RFftPlan1DY(mParameters.getFullDimensionSizes());

    // Z shifts
    if (mParameters.isSimulation3D())
    {
      CufftComplexMatrix::createR2CFftPlan1DZ(mParameters.getFullDimensionSizes());
      CufftComplexMatrix::createC2RFftPlan1DZ(mParameters.getFullDimensionSizes());
    }
  }// end u_non_staggered
}// end of initializeCufftPlans
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute pre-processing phase.
 */
template<SD simulationDimension>
void KSpaceFirstOrderSolver::preProcessing()
{
  // Get the correct sensor mask and recompute indices
  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    getIndexMatrix(MI::kSensorMaskIndex).recomputeIndicesToCPP();
  }

  if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners)
  {
    getIndexMatrix(MI::kSensorMaskCorners).recomputeIndicesToCPP();
  }

  if ((mParameters.getTransducerSourceFlag() != 0) ||
      (mParameters.getVelocityXSourceFlag() != 0)  ||
      (mParameters.getVelocityYSourceFlag() != 0)  ||
      (mParameters.getVelocityZSourceFlag() != 0)
     )
  {
    getIndexMatrix(MI::kVelocitySourceIndex).recomputeIndicesToCPP();
  }

  if (mParameters.getTransducerSourceFlag() != 0)
  {
    getIndexMatrix(MI::kDelayMask).recomputeIndicesToCPP();
  }

  if (mParameters.getPressureSourceFlag() != 0)
  {
    getIndexMatrix(MI::kPressureSourceIndex).recomputeIndicesToCPP();
  }

  // Compute dt / rho0_sg...
  if (!mParameters.getRho0ScalarFlag())
  { // Non-uniform grid cannot be pre-calculated :-(
    // rho is a matrix
    if (mParameters.getNonUniformGridFlag())
    {
      generateInitialDenisty<simulationDimension>();
    }
    else
    {
      getRealMatrix(MI::kDtRho0Sgx).scalarDividedBy(mParameters.getDt());
      getRealMatrix(MI::kDtRho0Sgy).scalarDividedBy(mParameters.getDt());
      if (simulationDimension == SD::k3D)
      {
        getRealMatrix(MI::kDtRho0Sgz).scalarDividedBy(mParameters.getDt());
      }
    }
  }

  // Generate constant matrices
  generateDerivativeOperators();

  // Generate absorption variables and kappa
  switch (mParameters.getAbsorbingFlag())
  {
    case Parameters::AbsorptionType::kLossless:
    {
      generateKappa();
      break;
    }
    case Parameters::AbsorptionType::kPowerLaw:
    {
      generateKappaAndNablas();
      generateTauAndEta();
      break;
    }
    case Parameters::AbsorptionType::kStokes:
    {
      generateKappa();
      generateTau();
      break;
    }
  }

  // Generate sourceKappa
  if (((mParameters.getVelocitySourceMode() == Parameters::SourceMode::kAdditive) ||
       (mParameters.getPressureSourceMode() == Parameters::SourceMode::kAdditive)) &&
      (mParameters.getPressureSourceFlag()  ||
       mParameters.getVelocityXSourceFlag() ||
       mParameters.getVelocityYSourceFlag() ||
       mParameters.getVelocityZSourceFlag()))
  {
    generateSourceKappa();
  }

  // Non-staggered shift variables
  if (mParameters.getStoreVelocityNonStaggeredRawFlag())
  {
    generateNonStaggeredShiftVariables();
  }

  // Generate PML
  generatePml();

  // Calculate c^2. It has to be after kappa gen... because of c modification
  generateC2();

}// end of preProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the main time loop of KSpaceFirstOrder solver.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void KSpaceFirstOrderSolver::computeMainLoop()
{
  // Templated CUDA solver class.
  using SolverKernels = SolverCudaKernels<simulationDimension,
                                          rho0ScalarFlag,
                                          bOnAScalarFlag,
                                          c0ScalarFlag,
                                          alphaCoefScalarFlag>;

  mActPercent = 0;
  // Set the actual progress percentage to correspond the time index after recovery
  if (mParameters.getTimeIndex() > 0)
  {
    // We're restarting after checkpoint
    mIsTimestepRightAfterRestore = true;
    mActPercent = (100 * mParameters.getTimeIndex()) / mParameters.getNt();
  }

  // Progress header
  Logger::log(Logger::LogLevel::kBasic,kOutFmtSimulationHeader);

  // Copy matrix data to the GPU
  mMatrixContainer.copyMatricesToDevice();
  // Copy matrix pointers to a GPU container
  mMatrixContainer.copyContainerToDeveice();

  mIterationTime.start();

  // Execute main loop
  while ((mParameters.getTimeIndex() < mParameters.getNt()) &&
         (!mParameters.isTimeToCheckpoint(mTotalTime)))
  {
    const size_t timeIndex = mParameters.getTimeIndex();

    // Compute velocity
    computeVelocity<simulationDimension, rho0ScalarFlag>();

    // Add in the velocity source term
    addVelocitySource();

    // Add in the transducer source term (t = t1) to ux
    if (mParameters.getTransducerSourceFlag() > timeIndex)
    {
      SolverKernels::addTransducerSource(mMatrixContainer);
    }

    // Compute gradient of velocity
    computeVelocityGradient<simulationDimension>();

    // Compute density
    if (mParameters.getNonLinearFlag())
    {
      SolverKernels::computeDensityNonlinear();
    }
    else
    {
      SolverKernels::computeDensityLinear();
    }

    // Add in the source pressure term
    addPressureSource<simulationDimension>();

    if (mParameters.getNonLinearFlag())
    {
      computePressureNonlinear<simulationDimension,
                               rho0ScalarFlag,
                               bOnAScalarFlag,
                               c0ScalarFlag,
                               alphaCoefScalarFlag>();
    }
    else
    {
      computePressureLinear<simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag, alphaCoefScalarFlag>();
    }

    // Calculate initial pressure
    if ((timeIndex == 0) && (mParameters.getInitialPressureSourceFlag() == 1))
    {
      addInitialPressureSource<simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag>();
    }

    storeSensorData();
    printStatistics();

    mParameters.incrementTimeIndex();
    mIsTimestepRightAfterRestore = false;
  }// Time loop

  // Since disk operations are one step delayed, we have to do the last one here.
  // However, we need to check if the loop wasn't skipped due to very short checkpoint interval
  if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && (!mIsTimestepRightAfterRestore))
  {
    mOutputStreamContainer.flushRawStreams();
  }
}// end of computeMainLoop
//----------------------------------------------------------------------------------------------------------------------

/**
 * Post processing the quantities, closing the output streams and storing the sensor mask.
 */
void KSpaceFirstOrderSolver::postProcessing()
{
  if (mParameters.getStorePressureFinalAllFlag())
  {
    RealMatrix& p = getRealMatrix(MI::kP);

    p.copyFromDevice();
    p.writeData(mParameters.getOutputFile(),
                mOutputStreamContainer.getStreamHdf5Name(OI::kFinalPressure),
                mParameters.getCompressionLevel());
  }// p_final

  if (mParameters.getStoreVelocityFinalAllFlag())
  {
    RealMatrix& uxSgx = getRealMatrix(MI::kUxSgx);
    RealMatrix& uySgy = getRealMatrix(MI::kUySgy);

    uxSgx.copyFromDevice();
    uySgy.copyFromDevice();

    uxSgx.writeData(mParameters.getOutputFile(),
                    mOutputStreamContainer.getStreamHdf5Name(OI::kFinalVelocityX),
                    mParameters.getCompressionLevel());
    uySgy.writeData(mParameters.getOutputFile(),
                    mOutputStreamContainer.getStreamHdf5Name(OI::kFinalVelocityY),
                    mParameters.getCompressionLevel());
    if (mParameters.isSimulation3D())
    {
      RealMatrix& uzSgz = getRealMatrix(MI::kUzSgz);;

      uzSgz.copyFromDevice();
      uzSgz.writeData(mParameters.getOutputFile(),
                      mOutputStreamContainer.getStreamHdf5Name(OI::kFinalVelocityZ),
                      mParameters.getCompressionLevel());
    }
  }// u_final

  // Apply post-processing, flush data on disk
  mOutputStreamContainer.postProcessStreams();
  mOutputStreamContainer.closeStreams();


  // Store sensor mask if wanted
  if (mParameters.getCopySensorMaskFlag())
  {
    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
    {
      IndexMatrix& sensorMask = getIndexMatrix(MI::kSensorMaskIndex);

      sensorMask.recomputeIndicesToMatlab();
      sensorMask.writeData(mParameters.getOutputFile(),
                           mMatrixContainer.getMatrixHdf5Name(MI::kSensorMaskIndex),
                           mParameters.getCompressionLevel());
    }

    if (mParameters.getSensorMaskType() == Parameters::SensorMaskType::kCorners)
    {
      IndexMatrix& sensorMask = getIndexMatrix(MI::kSensorMaskCorners);

      sensorMask.recomputeIndicesToMatlab();
      sensorMask.writeData(mParameters.getOutputFile(),
                           mMatrixContainer.getMatrixHdf5Name(MI::kSensorMaskCorners),
                           mParameters.getCompressionLevel());
    }
  }
}// end of postProcessing
//----------------------------------------------------------------------------------------------------------------------

/**
 * Store sensor data.
 */
void KSpaceFirstOrderSolver::storeSensorData()
{
  // Unless the time for sampling has come, exit.
  if (mParameters.getTimeIndex() >= mParameters.getSamplingStartTimeIndex())
  {
    // Read event for t_index - 1. If sampling did not occur by then, ignored it.
    // If it did store data on disk (flush) - the GPU is running asynchronously.
    // But be careful, flush has to be one step delayed to work correctly.
    // When restoring from checkpoint we have to skip the first flush.
    if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex() && !mIsTimestepRightAfterRestore)
    {
      mOutputStreamContainer.flushRawStreams();
    }

    // if --u_non_staggered is switched on, calculate unstaggered velocity.
    if (mParameters.getStoreVelocityNonStaggeredRawFlag())
    {
      computeShiftedVelocity();
    }

    // Sample data for step t  (store event for sampling in next turn)
    mOutputStreamContainer.sampleStreams();
    // The last step (or data after) checkpoint are flushed in the main loop
  }
}// end of storeSensorData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Write statistics and the header into the output file.
 */
void KSpaceFirstOrderSolver::writeOutputDataInfo()
{
  // Write timeIndex into the output file
  mParameters.getOutputFile().writeScalarValue(mParameters.getOutputFile().getRootGroup(),
                                               mParameters.getTimeIndexHdf5Name(),
                                               mParameters.getTimeIndex());

  // Write scalars
  mParameters.writeScalarsToOutputFile();
  Hdf5FileHeader& fileHeader = mParameters.getFileHeader();

  // Write File header
  fileHeader.setCodeName(getCodeName());
  fileHeader.setMajorFileVersion();
  fileHeader.setMinorFileVersion();
  fileHeader.setActualCreationTime();
  fileHeader.setFileType(Hdf5FileHeader::FileType::kOutput);
  fileHeader.setHostInfo();

  fileHeader.setMemoryConsumption(getHostMemoryUsage());

  // Stop total timer here
  mTotalTime.stop();
  fileHeader.setExecutionTimes(mTotalTime.getElapsedTimeOverAllLegs(),
                               mDataLoadTime.getElapsedTimeOverAllLegs(),
                               mPreProcessingTime.getElapsedTimeOverAllLegs(),
                               mSimulationTime.getElapsedTimeOverAllLegs(),
                               mPostProcessingTime.getElapsedTimeOverAllLegs());

  fileHeader.setNumberOfCores();
  fileHeader.writeHeaderToOutputFile(mParameters.getOutputFile());
}// end of writeOutputDataInfo
//----------------------------------------------------------------------------------------------------------------------

/**
 * Save checkpoint data into the checkpoint file, flush aggregated outputs into the output file.
 */
void KSpaceFirstOrderSolver::writeCheckpointData()
{
  // Create checkpoint file
  Hdf5File& checkpointFile = mParameters.getCheckpointFile();
  // If it happens and the file is opened (from the recovery, close it)
  if (checkpointFile.isOpen())
  {
    checkpointFile.close();
  }

  Logger::log(Logger::LogLevel::kFull, kOutFmtStoringCheckpointData);
  Logger::flush(Logger::LogLevel::kFull);

  // Create the new file (overwrite the old one)
  checkpointFile.create(mParameters.getCheckpointFileName());

  //------------------------------------------------ Store Matrices --------------------------------------------------//
  // Store all necessary matrices in checkpoint file
  mMatrixContainer.storeDataIntoCheckpointFile();

  // Write t_index
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(),
                                  mParameters.getTimeIndexHdf5Name(),
                                  mParameters.getTimeIndex());

  // Store basic dimension sizes (Nx, Ny, Nz) - Nt is not necessary
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(),
                                  mParameters.getNxHdf5Name(),
                                  mParameters.getFullDimensionSizes().nx);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(),
                                  mParameters.getNyHdf5Name(),
                                  mParameters.getFullDimensionSizes().ny);
  checkpointFile.writeScalarValue(checkpointFile.getRootGroup(),
                                  mParameters.getNzHdf5Name(),
                                  mParameters.getFullDimensionSizes().nz);

  // Write checkpoint file header
  Hdf5FileHeader fileHeader = mParameters.getFileHeader();

  fileHeader.setFileType(Hdf5FileHeader::FileType::kCheckpoint);
  fileHeader.setCodeName(getCodeName());
  fileHeader.setActualCreationTime();

  fileHeader.writeHeaderToCheckpointFile(checkpointFile);

  // Close the checkpoint file
  checkpointFile.close();
  Logger::log(Logger::LogLevel::kFull, kOutFmtDone);

  // Checkpoint output streams only if necessary (t_index > start_index), we're here one step ahead!
  if (mParameters.getTimeIndex() > mParameters.getSamplingStartTimeIndex())
  {
    Logger::log(Logger::LogLevel::kFull,kOutFmtStoringSensorData);
    Logger::flush(Logger::LogLevel::kFull);

    mOutputStreamContainer.checkpointStreams();

    Logger::log(Logger::LogLevel::kFull, kOutFmtDone);
  }

  mOutputStreamContainer.closeStreams();
}// end of writeCheckpointData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print progress statistics.
 */
void KSpaceFirstOrderSolver::printStatistics()
{
  const size_t nt        = mParameters.getNt();
  const size_t timeIndex = mParameters.getTimeIndex();

  if (timeIndex > (mActPercent * nt * 0.01f))
  {
    mActPercent += mParameters.getProgressPrintInterval();

    mIterationTime.stop();

    const double elTime         = mIterationTime.getElapsedTime();
    const double elTimeWithLegs = mIterationTime.getElapsedTime() + mSimulationTime.getElapsedTimeOverPreviousLegs();
    const double toGo           = ((elTimeWithLegs / double((timeIndex + 1)) * nt)) - elTimeWithLegs;

    struct tm* current = nullptr;
    time_t now;

    time(&now);
    now += toGo;
    current = localtime(&now);

    Logger::log(Logger::LogLevel::kBasic,
                kOutFmtSimulationProgress,
                size_t(((timeIndex) / (nt * 0.01f))),'%',
                elTime, toGo,
                current->tm_mday, current->tm_mon+1, current->tm_year-100,
                current->tm_hour, current->tm_min, current->tm_sec);
    Logger::flush(Logger::LogLevel::kBasic);
  }
}// end of printStatistics
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new values of acoustic velocity in all used dimensions (UxSgx, UySgy, UzSgz).
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag>
void KSpaceFirstOrderSolver::computeVelocity()
{
  using SolverKernels = SolverCudaKernels<simulationDimension, rho0ScalarFlag>;

  // fftn(p);
  getTempCufftX().computeR2CFftND(getRealMatrix(MI::kP));

  // bsxfun(@times, ddx_k_shift_pos, kappa .* pre_result) , for all 3 dims
  SolverKernels::computePressureGradient();

  // ifftn(pre_result)
  getTempCufftX().computeC2RFftND(getRealMatrix(MI::kTemp1RealND));
  getTempCufftY().computeC2RFftND(getRealMatrix(MI::kTemp2RealND));
  if (simulationDimension == SD::k3D)
  {
    getTempCufftZ().computeC2RFftND(getRealMatrix(MI::kTemp3RealND));
  }

  // bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* (pre_result))
  if (mParameters.getNonUniformGridFlag())
  {
    SolverKernels::computeVelocityHomogeneousNonuniform();
  }
  else
  {
    SolverKernels::computeVelocityUniform();
  }
}// end of computeVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculated shifted velocities.
 */
void KSpaceFirstOrderSolver::computeShiftedVelocity()
{
  // Templated CUDA solver class - we pick the simplest one since the computation is the same for all cases.
  using SolverKernels = SolverCudaKernels<>;

  CufftComplexMatrix& tempCufftShift = getTempCufftShift();

  // uxShifted
  tempCufftShift.computeR2CFft1DX(getRealMatrix(MI::kUxSgx));
  SolverKernels::computeVelocityShiftInX();
  tempCufftShift.computeC2RFft1DX(getRealMatrix(MI::kUxShifted));

  // uyShifted
  tempCufftShift.computeR2CFft1DY(getRealMatrix(MI::kUySgy));
  SolverKernels::computeVelocityShiftInY();
  tempCufftShift.computeC2RFft1DY(getRealMatrix(MI::kUyShifted));

  if (mParameters.isSimulation3D())
  {
    // uzShifted
    tempCufftShift.computeR2CFft1DZ(getRealMatrix(MI::kUzSgz));
    SolverKernels::computeVelocityShiftInZ();
    tempCufftShift.computeC2RFft1DZ(getRealMatrix(MI::kUzShifted));
  }
}// end of computeShiftedVelocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new gradient of velocity (duxdx, duydy, duzdz).
 */
template<SD simulationDimension>
void KSpaceFirstOrderSolver::computeVelocityGradient()
{
  using SolverKernels = SolverCudaKernels<simulationDimension>;

  getTempCufftX().computeR2CFftND(getRealMatrix(MI::kUxSgx));
  getTempCufftY().computeR2CFftND(getRealMatrix(MI::kUySgy));
  if (simulationDimension == SD::k3D)
  {
    getTempCufftZ().computeR2CFftND(getRealMatrix(MI::kUzSgz));
  }

  // Calculate velocity gradient on uniform grid
  SolverKernels::computeVelocityGradient();

  getTempCufftX().computeC2RFftND(getRealMatrix(MI::kDuxdx));
  getTempCufftY().computeC2RFftND(getRealMatrix(MI::kDuydy));
  if (simulationDimension == SD::k3D)
  {
    getTempCufftZ().computeC2RFftND(getRealMatrix(MI::kDuzdz));
  }

  // Non-uniform grid
  if (mParameters.getNonUniformGridFlag() != 0)
  {
    SolverKernels::computeVelocityGradientShiftNonuniform();
  }// non-uniform grid
}// end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute acoustic pressure for non-linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void KSpaceFirstOrderSolver::computePressureNonlinear()
{
  using SolverKernels = SolverCudaKernels<simulationDimension,
                                          rho0ScalarFlag,
                                          bOnAScalarFlag,
                                          c0ScalarFlag,
                                          alphaCoefScalarFlag>;

  switch (mParameters.getAbsorbingFlag())
  {
    case Parameters::AbsorptionType::kLossless:
    {
      SolverKernels::sumPressureNonlinearLossless();
      break;
    }

    case Parameters::AbsorptionType::kPowerLaw:
    {
      RealMatrix& densitySum          = getRealMatrix(MI::kTemp1RealND);
      RealMatrix& nonlinearTerm       = getRealMatrix(MI::kTemp2RealND);
      RealMatrix& velocityGradientSum = getRealMatrix(MI::kTemp3RealND);

      // Reusing of the temp variables
      RealMatrix& absorbTauTerm = velocityGradientSum;
      RealMatrix& absorbEtaTerm = densitySum;

      // Compute three temporary sums in the new pressure formula, non-linear absorbing case.
      SolverKernels::computePressureTermsNonlinearPowerLaw(densitySum, nonlinearTerm, velocityGradientSum);

      getTempCufftX().computeR2CFftND(velocityGradientSum);
      getTempCufftY().computeR2CFftND(densitySum);

      SolverKernels::computeAbsorbtionTerm();

      getTempCufftX().computeC2RFftND(absorbTauTerm);
      getTempCufftY().computeC2RFftND(absorbEtaTerm);

      SolverKernels::sumPressureTermsNonlinearPowerLaw(nonlinearTerm, absorbTauTerm, absorbEtaTerm);
      break;
    }

    case Parameters::AbsorptionType::kStokes:
    {
      SolverKernels::sumPressureNonlinearStokes();
      break;
    }
  }
}// end of computePressureNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute new p for linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void KSpaceFirstOrderSolver::computePressureLinear()
{
  using SolverKernels = SolverCudaKernels<simulationDimension,
                                          rho0ScalarFlag,
                                          bOnAScalarFlag,
                                          c0ScalarFlag,
                                          alphaCoefScalarFlag>;

  switch (mParameters.getAbsorbingFlag())
  {
    case Parameters::AbsorptionType::kLossless:
    {
      SolverKernels::sumPressureLinearLossless();
      break;
    }

    case Parameters::AbsorptionType::kPowerLaw:
    {
      RealMatrix& densitySum           = getRealMatrix(MI::kTemp1RealND);
      RealMatrix& velocityGradientTerm = getRealMatrix(MI::kTemp2RealND);

      RealMatrix& absorbTauTerm        = getRealMatrix(MI::kTemp2RealND);
      RealMatrix& absorbEtaTerm        = getRealMatrix(MI::kTemp3RealND);

      SolverKernels::computePressureTermsLinearPowerLaw(densitySum, velocityGradientTerm);

      // ifftn ( absorbNabla1 * fftn (rho0 * (duxdx + duydy + duzdz))
      getTempCufftX().computeR2CFftND(velocityGradientTerm);
      getTempCufftY().computeR2CFftND(densitySum);

      SolverKernels::computeAbsorbtionTerm();

      getTempCufftX().computeC2RFftND(absorbTauTerm);
      getTempCufftY().computeC2RFftND(absorbEtaTerm);

      SolverKernels::sumPressureTermsLinearPowerLaw(absorbTauTerm, absorbEtaTerm, densitySum);
      break;
    }

    case Parameters::AbsorptionType::kStokes:
    {
      SolverKernels::sumPressureLinearStokes();
      break;
    }
  }
}// end of computePressureLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add in pressure source.
 */
template<SD simulationDimension>
void KSpaceFirstOrderSolver::addPressureSource()
{
  // Templated CUDA solver class - nothing but simulation dimension is used
  using SolverKernels = SolverCudaKernels<simulationDimension>;

  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getPressureSourceFlag() > timeIndex)
  {
    if (mParameters.getPressureSourceMode() != Parameters::SourceMode::kAdditive)
    { // Executed Dirichlet and AdditiveNoCorrection source
      SolverKernels::addPressureSource(mMatrixContainer);
    }
    else
    { // Execute Additive source
      RealMatrix& scaledSource = getRealMatrix(MI::kTemp1RealND);

      scaleSource(scaledSource,
                  getRealMatrix(MI::kPressureSourceInput),
                  getIndexMatrix(MI::kPressureSourceIndex),
                  mParameters.getPressureSourceMany());

      // Insert source
      SolverKernels::addPressureScaledSource(scaledSource);
    }// Additive source
  }// apply source
}// end of AddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Add velocity source to the particle velocity.
 */
void KSpaceFirstOrderSolver::addVelocitySource()
{
  // Templated CUDA solver class - we pick the simplest one since the computation is the same for all cases.
  using SolverKernels = SolverCudaKernels<>;

  size_t timeIndex = mParameters.getTimeIndex();

  if (mParameters.getVelocitySourceMode() != Parameters::SourceMode::kAdditive)
  { // Executed Dirichlet and AdditiveNoCorrection source
    if (mParameters.getVelocityXSourceFlag() > timeIndex)
    {
      SolverKernels::addVelocitySource(getRealMatrix(MI::kUxSgx),
                                       getRealMatrix(MI::kVelocityXSourceInput),
                                       getIndexMatrix(MI::kVelocitySourceIndex));
    }
    if (mParameters.getVelocityYSourceFlag() > timeIndex)
    {
      SolverKernels::addVelocitySource(getRealMatrix(MI::kUySgy),
                                       getRealMatrix(MI::kVelocityYSourceInput),
                                       getIndexMatrix(MI::kVelocitySourceIndex));
    }

    if ((mParameters.isSimulation3D()) && (mParameters.getVelocityZSourceFlag() > timeIndex))
    {
      SolverKernels::addVelocitySource(getRealMatrix(MI::kUzSgz),
                                       getRealMatrix(MI::kVelocityZSourceInput),
                                       getIndexMatrix(MI::kVelocitySourceIndex));
    }
  }
  else
  { // Execute Additive source
    RealMatrix& scaledSource = getRealMatrix(MI::kTemp1RealND);

    if (mParameters.getVelocityXSourceFlag() > timeIndex)
    {
      scaleSource(scaledSource,
                  getRealMatrix(MI::kVelocityXSourceInput),
                  getIndexMatrix(MI::kVelocitySourceIndex),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverKernels::addVelocityScaledSource(getRealMatrix(MI::kUxSgx), scaledSource);
    }

    if (mParameters.getVelocityYSourceFlag() > timeIndex)
    {
      scaleSource(scaledSource,
                  getRealMatrix(MI::kVelocityYSourceInput),
                  getIndexMatrix(MI::kVelocitySourceIndex),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverKernels::addVelocityScaledSource(getRealMatrix(MI::kUySgy), scaledSource);
    }

    if ((mParameters.isSimulation3D()) && (mParameters.getVelocityZSourceFlag() > timeIndex))
    {
      scaleSource(scaledSource,
                  getRealMatrix(MI::kVelocityZSourceInput),
                  getIndexMatrix(MI::kVelocitySourceIndex),
                  mParameters.getVelocitySourceMany());

      // Insert source
      SolverKernels::addVelocityScaledSource(getRealMatrix(MI::kUzSgz), scaledSource);
    }
  }
}// end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Scale source signal.
 */
void KSpaceFirstOrderSolver::scaleSource(RealMatrix&        scaledSource,
                                         const RealMatrix&  sourceInput,
                                         const IndexMatrix& sourceIndex,
                                         const size_t       manyFlag)
{
  // Templated CUDA solver class - we pick the simplest one since the computation is the same for all cases.
  using SolverKernels = SolverCudaKernels<>;

  // Zero source scaling matrix on GPU.
  scaledSource.zeroDeviceMatrix();
  // Inject source to scaling matrix
  SolverKernels::insertSourceIntoScalingMatrix(scaledSource, sourceInput, sourceIndex, manyFlag);
  // Compute FFT
  getTempCufftX().computeR2CFftND(scaledSource);
  // Calculate gradient
  SolverKernels::computeSourceGradient(getTempCufftX(), getRealMatrix(MI::kSourceKappa));
  // Compute iFFT
  getTempCufftX().computeC2RFftND(scaledSource);
}// end of scaleSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate initial pressure source.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag>
void KSpaceFirstOrderSolver::addInitialPressureSource()
{
  // Templated CUDA solver class.
  using SolverKernels = SolverCudaKernels<simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag>;

  // Add the initial pressure to rho as a mass source
  SolverKernels::addInitialPressureSource();

  // Compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)
  // which forces u(t = t1) = 0
  getTempCufftX().computeR2CFftND(getRealMatrix(MI::kP));

  SolverKernels::computePressureGradient();

  getTempCufftX().computeC2RFftND(getRealMatrix(MI::kUxSgx));
  getTempCufftY().computeC2RFftND(getRealMatrix(MI::kUySgy));
  if (simulationDimension == SD::k3D)
  {
    getTempCufftZ().computeC2RFftND(getRealMatrix(MI::kUzSgz));
  }

  if (mParameters.getNonUniformGridFlag())
  { // Non-uniform grid, homogeneous
    SolverKernels::computeInitialVelocityHomogeneousNonuniform();
  }
  else
  { // Uniform grid, homogeneous
    SolverKernels::computeInitialVelocityUniform();
  }
}// end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Prepare dt./ rho0  for non-uniform grid.
 */
template<Parameters::SimulationDimension simulationDimension>
void KSpaceFirstOrderSolver::generateInitialDenisty()
{
  const DimensionSizes& dimensionSizes = mParameters.getFullDimensionSizes();

  const float dt = mParameters.getDt();

  const float* duxdxnSgx = getRealData(MI::kDxudxnSgx);
  const float* duydynSgy = getRealData(MI::kDyudynSgy);
  const float* duzdznSgz = getRealData(MI::kDzudznSgz, simulationDimension == SD::k3D);

  float* dtRho0Sgx = getRealData(MI::kDtRho0Sgx);
  float* dtRho0Sgy = getRealData(MI::kDtRho0Sgy);
  float* dtRho0Sgz = getRealData(MI::kDtRho0Sgz, simulationDimension == SD::k3D);

  #pragma omp parallel for schedule(static) if (simulationDimension == SD::k3D)
  for (size_t z = 0; z < dimensionSizes.nz; z++)
  {
    #pragma omp parallel for schedule(static) if (simulationDimension == SD::k2D)
    for (size_t y = 0; y < dimensionSizes.ny; y++)
    {
      for (size_t x = 0; x < dimensionSizes.nx; x++)
      {
        const size_t i = get1DIndex(z, y, x, dimensionSizes);

        dtRho0Sgx[i] = (dt * duxdxnSgx[x]) / dtRho0Sgx[i];
        dtRho0Sgy[i] = (dt * duydynSgy[y]) / dtRho0Sgy[i];
        if (simulationDimension == SD::k3D)
        {
          dtRho0Sgz[i] = (dt * duzdznSgz[z]) / dtRho0Sgz[i];
        }
      }// x
    }// y
  }// z
}// end of generateInitialDenisty
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate kappa matrix for lossless medium.
 * For 2D simulation, the zPart == 0.
 */
void KSpaceFirstOrderSolver::generateKappa()
{
  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  // For 2D simulation set dz to 0
  const float dz2Rec = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * float(M_PI);

  const float nxRec = 1.0f / float(mParameters.getFullDimensionSizes().nx);
  const float nyRec = 1.0f / float(mParameters.getFullDimensionSizes().ny);
  // For 2D simulation, nzRec remains 1
  const float nzRec = 1.0f / float(mParameters.getFullDimensionSizes().nz);

  float* kappa = getRealData(MI::kKappa);

  // Generate wave number in given direction
  auto kPart = [](float i, float sizeRec, float dispRec)
  {
    float k = 0.5f - fabs(0.5f - i * sizeRec);
    return (k * k) * dispRec;
  };// end of kPart

  #pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++)
  {
    const float kz = kPart(float(z), nzRec, dz2Rec);

    #pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++)
    {
      const float ky = kPart(float(y), nyRec, dy2Rec);

      for (size_t x = 0; x < reducedDimensionSizes.nx; x++)
      {
        const size_t i = get1DIndex(z, y, x, reducedDimensionSizes);

        const float kx = kPart(float(x), nxRec, dx2Rec);
        const float k  = cRefDtPi * sqrt(kx + ky + kz);

        // kappa element
        kappa[i] = (k == 0.0f) ? 1.0f : sin(k) / k;
      }// x
    }// y
  }// z
}// end of generateKappa
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate derivative operators (dd{x,y,z}_k_shift_pos, dd{x,y,z}_k_shift_neg).
 */
void KSpaceFirstOrderSolver::generateDerivativeOperators()
{
  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();
  const DimensionSizes& dimensionSizes        = mParameters.getFullDimensionSizes();

  constexpr FloatComplex imagUnit = FloatComplex(0, 1);
  constexpr FloatComplex posExp   = FloatComplex(0, 1);
  constexpr FloatComplex negExp   = FloatComplex(0,-1);
  constexpr float        pi2      = 2.0f * float(M_PI);

  const float dx = mParameters.getDx();
  const float dy = mParameters.getDy();
  const float dz = (mParameters.isSimulation3D()) ? mParameters.getDz() : 0.0f;

  FloatComplex* ddxKShiftPos = getComplexData(MI::kDdxKShiftPosR);
  FloatComplex* ddyKShiftPos = getComplexData(MI::kDdyKShiftPos);
  FloatComplex* ddzKShiftPos = getComplexData(MI::kDdzKShiftPos, mParameters.isSimulation3D());

  FloatComplex* ddxKShiftNeg = getComplexData(MI::kDdxKShiftNegR);
  FloatComplex* ddyKShiftNeg = getComplexData(MI::kDdyKShiftNeg);
  FloatComplex* ddzKShiftNeg = getComplexData(MI::kDdzKShiftNeg, mParameters.isSimulation3D());

  // Calculate ifft shift
  auto iFftShift = [](ptrdiff_t i, ptrdiff_t size)
  {
    return (i + (size / 2)) % size - (size / 2);
  };// end of iFftShift

  // Calculation done sequentially because the size of the arrays are small < 512
  // Moreover, there's a bug in Intel compiler under windows generating clobbered data.
  // ddxKShiftPos, ddxKShiftPos
  for (size_t i = 0; i < reducedDimensionSizes.nx; i++)
  {
    const ptrdiff_t shift    = iFftShift(i, dimensionSizes.nx);
    const float     kx       = (pi2 / dx) * (float(shift) / float(dimensionSizes.nx));
    const float     exponent = kx * dx * 0.5f;

    ddxKShiftPos[i] = imagUnit * kx * std::exp(posExp * exponent);
    ddxKShiftNeg[i] = imagUnit * kx * std::exp(negExp * exponent);
  }

  // ddyKShiftPos, ddyKShiftPos
  for (size_t i = 0; i < dimensionSizes.ny; i++)
  {
    const ptrdiff_t shift    = iFftShift(i, dimensionSizes.ny);
    const float     ky       = (pi2 / dy) * (float(shift) / float(dimensionSizes.ny));
    const float     exponent = ky * dy * 0.5f;

    ddyKShiftPos[i] = imagUnit * ky * std::exp(posExp * exponent);
    ddyKShiftNeg[i] = imagUnit * ky * std::exp(negExp * exponent);
  }

  // ddzKShiftPos, ddzKShiftNeg
  if (mParameters.isSimulation3D())
  {
    for (size_t i = 0; i < dimensionSizes.nz; i++)
    {
      const ptrdiff_t shift    = iFftShift(i, dimensionSizes.nz);
      const float     kz       = (pi2 / dz) * (float(shift) / float(dimensionSizes.nz));
      const float     exponent = kz * dz * 0.5f;

      ddzKShiftPos[i] = imagUnit * kz * std::exp(posExp * exponent);
      ddzKShiftNeg[i] = imagUnit * kz * std::exp(negExp * exponent);
    }
  }
}// end of generateDerivativeOperators
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate sourceKappa matrix for additive sources.
 * For 2D simulation, the zPart == 0.
 */
void KSpaceFirstOrderSolver::generateSourceKappa()
{
  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  const float dx2Rec = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dy2Rec = 1.0f / (mParameters.getDy() * mParameters.getDy());
  const float dz2Rec = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDtPi = mParameters.getCRef() * mParameters.getDt() * float(M_PI);

  const float nxRec = 1.0f / float(mParameters.getFullDimensionSizes().nx);
  const float nyRec = 1.0f / float(mParameters.getFullDimensionSizes().ny);
  const float nzRec = 1.0f / float(mParameters.getFullDimensionSizes().nz);

  float* sourceKappa = getRealData(MI::kSourceKappa);

  // Generate wave number in a given direction
  auto kPart = [](float i, float sizeRec, float dispRec)
  {
    float k = 0.5f - fabs(0.5f - i * sizeRec);
    return (k * k) * dispRec;
  };// end of kPart

  #pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++)
  {
    const float kz = kPart(float(z), nzRec, dz2Rec);

    #pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++)
    {
      const float ky = kPart(float(y), nyRec, dy2Rec);

      for (size_t x = 0; x < reducedDimensionSizes.nx; x++)
      {
        const size_t i = get1DIndex(z, y, x, reducedDimensionSizes);

        const float kx = kPart(float(x), nxRec, dx2Rec);

        // sourceKappa element
        sourceKappa[i] = cos(cRefDtPi * sqrt(kx + ky + kz));
      }// x
    }// y
  }// z
}// end of generateSourceKappa
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate kappa matrix, absorbNabla1, absorbNabla2 for absorbing medium.
 * For the 2D simulation the zPart == 0
 */
void KSpaceFirstOrderSolver::generateKappaAndNablas()
{
  const DimensionSizes& reducedDimensionSizes = mParameters.getReducedDimensionSizes();

  const float dxSqRec    = 1.0f / (mParameters.getDx() * mParameters.getDx());
  const float dySqRec    = 1.0f / (mParameters.getDy() * mParameters.getDy());
  const float dzSqRec    = (mParameters.isSimulation3D()) ? 1.0f / (mParameters.getDz() * mParameters.getDz()) : 0.0f;

  const float cRefDt2    = mParameters.getCRef() * mParameters.getDt() * 0.5f;
  const float pi2        = float(M_PI) * 2.0f;

  const size_t nx        = mParameters.getFullDimensionSizes().nx;
  const size_t ny        = mParameters.getFullDimensionSizes().ny;
  const size_t nz        = mParameters.getFullDimensionSizes().nz;

  const float nxRec      = 1.0f / float(nx);
  const float nyRec      = 1.0f / float(ny);
  const float nzRec      = 1.0f / float(nz);

  const float alphaPower = mParameters.getAlphaPower();

  float* kappa           = getRealData(MI::kKappa);
  float* absorbNabla1    = getRealData(MI::kAbsorbNabla1);
  float* absorbNabla2    = getRealData(MI::kAbsorbNabla2);

  // Generated wave number in a given direction
  auto kPart = [](float i, float sizeRec, float dispRec)
  {
    float k = 0.5f - fabs(0.5f - i * sizeRec);
    return (k * k) * dispRec;
  };// end of kPart

  #pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
  for (size_t z = 0; z < reducedDimensionSizes.nz; z++)
  {
    const float kz = kPart(float(z), nzRec, dzSqRec);

    #pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
    for (size_t y = 0; y < reducedDimensionSizes.ny; y++)
    {
      const float ky = kPart(float(y), nyRec, dySqRec);

      for (size_t x = 0; x < reducedDimensionSizes.nx; x++)
      {
        const float kx    = kPart(float(x), nxRec, dxSqRec);
        const float k     = pi2 * sqrt(kx + ky + kz);
        const float cRefK = cRefDt2 * k;

        const size_t i  = get1DIndex(z, y, x, reducedDimensionSizes);

        kappa[i]        = (cRefK == 0.0f) ? 1.0f : sin(cRefK) / cRefK;

        absorbNabla1[i] = pow(k, alphaPower - 2.0f);
        absorbNabla2[i] = pow(k, alphaPower - 1.0f);

        if (absorbNabla1[i] ==  std::numeric_limits<float>::infinity()) absorbNabla1[i] = 0.0f;
        if (absorbNabla2[i] ==  std::numeric_limits<float>::infinity()) absorbNabla2[i] = 0.0f;
      }// x
    }// y
  }// z
}// end of generateKappaAndNablas
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate absorbTau and absorbEta in for heterogenous medium.
 */
void KSpaceFirstOrderSolver::generateTauAndEta()
{
  if ((mParameters.getAlphaCoeffScalarFlag()) && (mParameters.getC0ScalarFlag()))
  { // Scalar values
    const float alphaPower       = mParameters.getAlphaPower();
    const float tanPi2AlphaPower = tan(float(M_PI_2) * alphaPower);
    const float alphaNeperCoeff  = (100.0f * pow(1.0e-6f / (2.0f * float(M_PI)), alphaPower)) /
                                   (20.0f * float(M_LOG10E));

    const float alphaCoeff2      = 2.0f * mParameters.getAlphaCoeffScalar() * alphaNeperCoeff;

    mParameters.setAbsorbTauScalar((-alphaCoeff2) * pow(mParameters.getC0Scalar(), alphaPower - 1.0f));
    mParameters.setAbsorbEtaScalar(  alphaCoeff2  * pow(mParameters.getC0Scalar(), alphaPower) * tanPi2AlphaPower);
  }
  else
  { // Matrix values
    const DimensionSizes& dimensionSizes = mParameters.getFullDimensionSizes();

    const bool   alphaCoeffScalarFlag = mParameters.getAlphaCoeffScalarFlag();
    const float  alphaCoeffScalar     = (alphaCoeffScalarFlag) ? mParameters.getAlphaCoeffScalar() : 0.0f;
    const float* alphaCoeffMatrix     = (alphaCoeffScalarFlag) ? nullptr : getRealData(MI::kTemp1RealND);

    const float  alphaPower       = mParameters.getAlphaPower();
    const float  tanPi2AlphaPower = tan(float(M_PI_2) * alphaPower);

    const bool   c0ScalarFlag = mParameters.getC0ScalarFlag();
    const float  c0Scalar     = (c0ScalarFlag) ? mParameters.getC0Scalar() : 0.0f;
    // Here c2 still holds just c0!
    const float* cOMatrix     = (c0ScalarFlag) ? nullptr : getRealData(MI::kC2);

    // alpha = 100 * alpha .* (1e-6 / (2 * pi)).^y ./
    //                        (20 * log10(exp(1)));
    const float alphaNeperCoeff = (100.0f * pow(1.0e-6f / (2.0f * float(M_PI)), alphaPower)) /
                                  (20.0f * float(M_LOG10E));

    float* absorbTau = getRealData(MI::kAbsorbTau);
    float* absorbEta = getRealData(MI::kAbsorbEta);

    #pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
    for (size_t z = 0; z < dimensionSizes.nz; z++)
    {
      #pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
      for (size_t y = 0; y < dimensionSizes.ny; y++)
      {
        for (size_t x = 0; x < dimensionSizes.nx; x++)
        {
          const size_t i = get1DIndex(z, y, x, dimensionSizes);

          const float alphaCoeff2 = 2.0f * alphaNeperCoeff *
                                    ((alphaCoeffScalarFlag) ? alphaCoeffScalar : alphaCoeffMatrix[i]);

          absorbTau[i] = (-alphaCoeff2) * pow(((c0ScalarFlag) ? c0Scalar : cOMatrix[i]), alphaPower - 1.0f);
          absorbEta[i] =   alphaCoeff2  * pow(((c0ScalarFlag) ? c0Scalar : cOMatrix[i]), alphaPower) * tanPi2AlphaPower;
        }// x
      }// y
    }// z
  }// matrix
}// end of generateTauAndEta
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate absorbTau for heterogenous medium with stokes absorption.
 */
void KSpaceFirstOrderSolver::generateTau()
{
  if ((mParameters.getAlphaCoeffScalarFlag()) && (mParameters.getC0ScalarFlag()))
  { // Scalar values
    const float alphaPower       = mParameters.getAlphaPower();
    const float alphaNeperCoeff  = (100.0f * pow(1.0e-6f / (2.0f * float(M_PI)), alphaPower)) /
                                   (20.0f * float(M_LOG10E));

    const float alphaCoeff2      = 2.0f * mParameters.getAlphaCoeffScalar() * alphaNeperCoeff;

    mParameters.setAbsorbTauScalar((-alphaCoeff2) * pow(mParameters.getC0Scalar(), alphaPower - 1.0f));
  }
  else
  { // Matrix values
    const DimensionSizes& dimensionSizes = mParameters.getFullDimensionSizes();

    const bool   alphaCoeffScalarFlag = mParameters.getAlphaCoeffScalarFlag();
    const float  alphaCoeffScalar     = (alphaCoeffScalarFlag) ? mParameters.getAlphaCoeffScalar() : 0.0f;
    // temp1 matrix holds alpha coeff matrix (used only once).
    const float* alphaCoeffMatrix     = (alphaCoeffScalarFlag) ? nullptr : getRealData(MI::kTemp1RealND);

    const bool   c0ScalarFlag = mParameters.getC0ScalarFlag();
    const float  c0Scalar     = (c0ScalarFlag) ? mParameters.getC0Scalar() : 0.0f;
    // Here c2 still holds just c0!
    const float* cOMatrix     = (c0ScalarFlag) ? nullptr : getRealData(MI::kC2);

    const float  alphaPower   = mParameters.getAlphaPower();

    // alpha = 100 * alpha .* (1e-6 / (2 * pi)).^y ./
    //                        (20 * log10(exp(1)));
    const float alphaNeperCoeff = (100.0f * pow(1.0e-6f / (2.0f * float(M_PI)), alphaPower)) /
                                  (20.0f * float(M_LOG10E));

    float* absorbTau = getRealData(MI::kAbsorbTau);

    #pragma omp parallel for schedule(static) if (mParameters.isSimulation3D())
    for (size_t z = 0; z < dimensionSizes.nz; z++)
    {
      #pragma omp parallel for schedule(static) if (mParameters.isSimulation2D())
      for (size_t y = 0; y < dimensionSizes.ny; y++)
      {
        #pragma omp simd
        for (size_t x = 0; x < dimensionSizes.nx; x++)
        {
          const size_t i = get1DIndex(z, y, x, dimensionSizes);

          const float alphaCoeff2 = 2.0f * alphaNeperCoeff *
                                    ((alphaCoeffScalarFlag) ? alphaCoeffScalar : alphaCoeffMatrix[i]);

          absorbTau[i] = (-alphaCoeff2) * pow(((c0ScalarFlag) ? c0Scalar : cOMatrix[i]), alphaPower - 1.0f);
        }// x
      }// y
    }// z
  }// matrix
}// end of generateTau
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate shift variables for non-staggered velocity sampling (x_shift_neg_r, y_shift_neg_r, z_shift_neg_r).
 */
void KSpaceFirstOrderSolver::generateNonStaggeredShiftVariables()
{
  const DimensionSizes dimensionSizes = mParameters.getFullDimensionSizes();
  const DimensionSizes shiftDimensions(dimensionSizes.nx / 2 + 1,
                                       dimensionSizes.ny / 2 + 1,
                                       dimensionSizes.nz / 2 + 1);

  constexpr FloatComplex negExp = FloatComplex(0,-1);
  constexpr float        pi2    = 2.0f * float(M_PI);

  const float dx = mParameters.getDx();
  const float dy = mParameters.getDy();
  const float dz = mParameters.getDz();

  FloatComplex* xShiftNeg = getComplexData(MI::kXShiftNegR);
  FloatComplex* yShiftNeg = getComplexData(MI::kYShiftNegR);
  FloatComplex* zShiftNeg = getComplexData(MI::kZShiftNegR, mParameters.isSimulation3D());

  // Calculate ifft shift
  auto iFftShift = [](ptrdiff_t i, ptrdiff_t size)
  {
    return (i + (size / 2)) % size - (size / 2);
  };// end of iFftShift

  // Calculation done sequentially because the size of the arrays are small < 512
  // Moreover, there's a bug in Intel compiler under windows generating clobbered data.
  // xShiftNeg - No SIMD due to Intel Compiler bug under Windows
  for (size_t i = 0; i < shiftDimensions.nx; i++)
  {
    const ptrdiff_t shift = iFftShift(i, dimensionSizes.nx);
    const float     kx    = (pi2 / dx) * (float(shift) / float(dimensionSizes.nx));

    xShiftNeg[i] = std::exp(negExp * kx * dx * 0.5f);
  }

  // yShiftNeg
  for (size_t i = 0; i < shiftDimensions.ny; i++)
  {
    const ptrdiff_t shift = iFftShift(i, dimensionSizes.ny);
    const float     ky    = (pi2 / dy) * (float(shift) / float(dimensionSizes.ny));

    yShiftNeg[i] = std::exp(negExp * ky * dy * 0.5f);
  }

  // zShiftNeg
  if (mParameters.isSimulation3D())
  {
    for (size_t i = 0; i < shiftDimensions.nz; i++)
    {
      const ptrdiff_t shift = iFftShift(i, dimensionSizes.nz);
      const float     kz    = (pi2 / dz) * (float(shift) / float(dimensionSizes.nz));

      zShiftNeg[i] = std::exp(negExp * kz * dz * 0.5f);
    }
  }
}// end of generateNonStaggeredShiftVariables
//----------------------------------------------------------------------------------------------------------------------

/**
 * Generate PML and staggered PML.
 */
void KSpaceFirstOrderSolver::generatePml()
{
  const DimensionSizes dimensionSizes = mParameters.getFullDimensionSizes();

  const float pmlXAlpha = mParameters.getPmlXAlpha();
  const float pmlYAlpha = mParameters.getPmlYAlpha();

  const size_t pmlXSize = mParameters.getPmlXSize();
  const size_t pmlYSize = mParameters.getPmlYSize();

  const float  cRefDx   = mParameters.getCRef() / mParameters.getDx();
  const float  cRefDy   = mParameters.getCRef() / mParameters.getDy();

  const float  dt2      = mParameters.getDt() * 0.5f;

  float* pmlX    = getRealData(MI::kPmlX);
  float* pmlY    = getRealData(MI::kPmlY);

  float* pmlXSgx = getRealData(MI::kPmlXSgx);
  float* pmlYSgy = getRealData(MI::kPmlYSgy);

  // Init arrays
  auto initPml = [](float* pml, float* pmlSg, size_t size)
  {
    for (size_t i = 0; i < size; i++)
    {
      pml[i]   = 1.0f;
      pmlSg[i] = 1.0f;
    }
  };// end of initPml

  // Calculate left value of PML exponent, for staggered use i + 0.5f, i shifted by -1 (Matlab indexing).
  auto pmlLeft = [dt2](float i, float cRef, float pmlAlpha, float pmlSize)
  {
    return exp(-dt2 * pmlAlpha * cRef * pow((i - pmlSize) / (-pmlSize), 4));
  };// end of pmlLeft.

  // Calculate right value of PML exponent, for staggered use i + 0.5f, i shifted by +1 (Matlab indexing).
  auto pmlRight = [dt2](float i, float cRef, float pmlAlpha, float pmlSize)
  {
    return exp(-dt2 * pmlAlpha * cRef * pow((i + 1.0f)/ pmlSize, 4));
  };// end of pmlRight.


  // PML in x dimension
  initPml(pmlX, pmlXSgx, dimensionSizes.nx);

  // Too difficult for SIMD
  for (size_t i = 0; i < pmlXSize; i++)
  {
    pmlX[i]    = pmlLeft(float(i),        cRefDx, pmlXAlpha, pmlXSize);
    pmlXSgx[i] = pmlLeft(float(i) + 0.5f, cRefDx, pmlXAlpha, pmlXSize);

    const size_t iR = dimensionSizes.nx - pmlXSize + i;

    pmlX[iR]    = pmlRight(float(i),        cRefDx, pmlXAlpha, pmlXSize);
    pmlXSgx[iR] = pmlRight(float(i) + 0.5f, cRefDx, pmlXAlpha, pmlXSize);
  }

  // PML in y dimension
  initPml(pmlY, pmlYSgy, dimensionSizes.ny);

  // Too difficult for SIMD
  for (size_t i = 0; i < pmlYSize; i++)
  {
    pmlY[i]    = pmlLeft(float(i),        cRefDy, pmlYAlpha, pmlYSize);
    pmlYSgy[i] = pmlLeft(float(i) + 0.5f, cRefDy, pmlYAlpha, pmlYSize);

    const size_t iR = dimensionSizes.ny - pmlYSize + i;

    pmlY[iR]    = pmlRight(float(i),        cRefDy, pmlYAlpha, pmlYSize);
    pmlYSgy[iR] = pmlRight(float(i) + 0.5f, cRefDy, pmlYAlpha, pmlYSize);
  }

  // PML in z dimension
  if (mParameters.isSimulation3D())
  {
    const float  pmlZAlpha = mParameters.getPmlZAlpha();
    const size_t pmlZSize  = mParameters.getPmlZSize();
    const float  cRefDz    = mParameters.getCRef() / mParameters.getDz();

    float* pmlZ    = getRealData(MI::kPmlZ);
    float* pmlZSgz = getRealData(MI::kPmlZSgz);

    initPml(pmlZ, pmlZSgz, dimensionSizes.nz);

    // Too difficult for SIMD
    for (size_t i = 0; i < pmlZSize; i++)
    {
      pmlZ[i]    = pmlLeft(float(i)       , cRefDz, pmlZAlpha, pmlZSize);
      pmlZSgz[i] = pmlLeft(float(i) + 0.5f, cRefDz, pmlZAlpha, pmlZSize);

      const size_t iR = dimensionSizes.nz - pmlZSize + i;

      pmlZ[iR]    = pmlRight(float(i),        cRefDz, pmlZAlpha, pmlZSize);
      pmlZSgz[iR] = pmlRight(float(i) + 0.5f, cRefDz, pmlZAlpha, pmlZSize);
    }
  }
}// end of generatePml
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute c^2 on the CPU side.
 */
void KSpaceFirstOrderSolver::generateC2()
{
  if (!mParameters.getC0ScalarFlag())
  { // Matrix values
    const size_t nElements = mParameters.getFullDimensionSizes().nElements();

    float* c2 = getRealData(MI::kC2);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nElements; i++)
    {
      c2[i] = c2[i] * c2[i];
    }
  }// matrix
}// generateC2
//----------------------------------------------------------------------------------------------------------------------

inline size_t KSpaceFirstOrderSolver::get1DIndex(const size_t          z,
                                                 const size_t          y,
                                                 const size_t          x,
                                                 const DimensionSizes& dimensionSizes) const
{
  return (z * dimensionSizes.ny + y) * dimensionSizes.nx + x;
}// end of get1DIndex
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

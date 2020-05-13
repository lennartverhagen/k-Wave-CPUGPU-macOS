/**
 * @file      CudaParameters.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for the class for setting CUDA kernel parameters.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 November  2015, 16:49 (created) \n
 *            11 February  2020, 16:21 (revised)
 *
 * @copyright Copyright (C) 2015 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

#include <Parameters/CudaParameters.h>
#include <Parameters/CudaDeviceConstants.cuh>
#include <Parameters/Parameters.h>

#include <Logger/Logger.h>

#include <KSpaceSolver/SolverCudaKernels.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
CudaParameters::CudaParameters()
  : mDeviceIdx(kDefaultDeviceIdx),
    mSolverBlockSize1D(kUndefinedSize),
    mSolverGridSize1D(kUndefinedSize),
    mSolverTransposeBlockSize(kUndefinedSize),
    mSolverTransposeGirdSize(kUndefinedSize),
    mSamplerBlockSize1D(kUndefinedSize),
    mSamplerGridSize1D(kUndefinedSize),
    mDeviceProperties()
{

}// end of CudaParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * Return the name of device used.
 */
std::string CudaParameters::getDeviceName() const
{
  if (strcmp(mDeviceProperties.name, "") == 0)
  {
    return kOutFmtDeviceNameUnknown;
  }
  return mDeviceProperties.name;
}// end of getDeviceName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select cuda device for execution. If no device is specified, the first free is chosen. The routine also checks
 * whether the CUDA runtime and driver version match and whether the GPU is supported by the code. If there is no
 * free device is present, the code terminates with a runtime error.
 */
void CudaParameters::selectDevice(const int deviceIdx)
{
  // Check CUDA driver version and if not sufficient, terminate
  checkCudaVersion();

  mDeviceIdx = deviceIdx;

  // Choose the GPU device with the most global memory
  int nDevices;
  cudaCheckErrors(cudaGetDeviceCount(&nDevices));
  cudaGetLastError();

  cudaError_t lastError;
  // If the user does not provided a specific GPU, use the first one
  if (deviceIdx == kDefaultDeviceIdx)
  {
    bool deviceFound = false;

    for (int testDevice = 0; testDevice < nDevices; testDevice++)
    {
      // Try to set the GPU and reset it
      cudaSetDevice(testDevice);
      cudaDeviceReset();
      lastError = cudaGetLastError();

      // Reset was done properly, test CUDA code version
      if (lastError == cudaSuccess)
      {
        // Read the GPU SM version and the kernel version
        bool cudaCodeVersionOK = checkCudaCodeVersion();
        lastError = cudaGetLastError();

        if (cudaCodeVersionOK && (lastError == cudaSuccess))
        {
          // Acquire the GPU
          mDeviceIdx = testDevice;
          deviceFound = true;
          break;
        }
      }
      // GPU was busy, reset and continue
      lastError = cudaDeviceReset();

      // Clear last error
      cudaGetLastError();
    }

    if (!deviceFound)
    {
      throw std::runtime_error(kErrFmtNoFreeDevice);
    }
  }
  else // Select the device the user wants
  {
    // Check if the specified device is acceptable - not busy, input parameter not out of bounds
    if ((mDeviceIdx > nDevices - 1) || (mDeviceIdx < 0))
    {
      throw std::runtime_error(Logger::formatMessage(kErrFmtBadDeviceIndex, mDeviceIdx, nDevices-1));
     }

    // Set the device and copy it's properties
    cudaSetDevice(mDeviceIdx);
    cudaDeviceReset();
    lastError = cudaGetLastError();

    bool cudaCodeVersionOK = checkCudaCodeVersion();
    lastError = cudaGetLastError();

    if ((lastError != cudaSuccess) || (!cudaCodeVersionOK))
    {
      lastError = cudaDeviceReset();

      throw std::runtime_error(Logger::formatMessage(kErrFmtDeviceIsBusy, mDeviceIdx));
    }
  }

  // Read the device that was allocated. No device synchronization can be used for error handling.
  cudaCheckErrorsNoSync(cudaGetDevice(&mDeviceIdx));
  cudaCheckErrorsNoSync(cudaGetLastError());

  // Reset the device to be able to set the flags. No device synchronization can be used for error handling.
  cudaCheckErrorsNoSync(cudaDeviceReset());
  cudaCheckErrorsNoSync(cudaGetLastError());

  // Enable mapped memory. No device synchronization can be used for error handling.
  cudaCheckErrorsNoSync(cudaSetDeviceFlags(cudaDeviceMapHost));

  // Get Device name. No device synchronization can be used for error handling.
  cudaCheckErrorsNoSync(cudaGetDeviceProperties(&mDeviceProperties, mDeviceIdx));

  // Check the GPU version. No device synchronization can be used for error handling.
  if (!checkCudaCodeVersion())
  {
    throw std::runtime_error(Logger::formatMessage(kErrFmtDeviceNotSupported, mDeviceIdx));
  }
}// end of selectDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Based on the dimension sizes, sensors masks, and the GPU architecture, adequate CUDA kernel configurations
 * are selected. \n
 *
 * @li <b>1D solver kernels</b> have blocks of size 256 threads and at least 8 times more blocks than SM processors.
 * This gives us full potential on all known GPUs and still not compromising the maximum number of blocks
 * and threads. \n
 *
 * @li <b> Transposition kernels </b> work by processing tiles of 32x32 by 4 warps. Every block is responsible for
 * one 2D slab. Block size for the transposition kernels (only 128 threads). \n
 *
 * @li <b> Streaming kernels </b>  always use 1D kernels of 256 threads and at least 8 times more blocks than SM
 * processors. The size of the grid is only tuned for linear sensor mask, since in this execution phase, we don't know
 * how many elements there are in the cuboid sensor mask. \n
 */
void CudaParameters::setKernelConfiguration()
{
  const Parameters& params = Parameters::getInstance();

  DimensionSizes fullDims(params.getFullDimensionSizes());

  // Set kernel configuration for 1D kernels
  mSolverBlockSize1D = 256;
  // Grid size is calculated based on the number of SM processors
  mSolverGridSize1D  = mDeviceProperties.multiProcessorCount * 8;

  // The grid size is to small, get 1 grid point per thread
  if ((size_t(mSolverGridSize1D) * size_t(mSolverBlockSize1D)) > fullDims.nElements())
  {
    mSolverGridSize1D  = int((fullDims.nElements()  + size_t(mSolverBlockSize1D) - 1 ) / size_t(mSolverBlockSize1D));
  }

  // Transposition kernels.
  mSolverTransposeBlockSize = dim3(32, 4 , 1);
  // Grid size for the transposition kernels
  mSolverTransposeGirdSize = dim3(mDeviceProperties.multiProcessorCount * 16, 1, 1);


  // Set configuration for Streaming kernels.
  mSamplerBlockSize1D = 256;

  mSamplerGridSize1D  = mDeviceProperties.multiProcessorCount * 8;

  // Tune number of blocks for index based sensor mask
  if (params.getSensorMaskType() == Parameters::SensorMaskType::kIndex)
  {
    // The sensor mask is smaller than 2048 * SMs than use a smaller number of blocks
    if ((size_t(mSamplerGridSize1D) * size_t(mSamplerBlockSize1D)) > params.getSensorMaskIndexSize())
    {
      mSamplerGridSize1D  = int((params.getSensorMaskIndexSize()  + size_t(mSamplerBlockSize1D) - 1)
                                / size_t(mSamplerBlockSize1D));
    }
  }
}// end of setKernelConfiguration
//----------------------------------------------------------------------------------------------------------------------

/**
 * Upload useful simulation constants into device constant memory.
 */
void CudaParameters::setUpDeviceConstants() const
{
  CudaDeviceConstants constantsToTransfer;

  Parameters& params = Parameters::getInstance();
  DimensionSizes fullDimSizes    = params.getFullDimensionSizes();
  DimensionSizes reducedDimSizes = params.getReducedDimensionSizes();


  // Set values for constant memory
  constantsToTransfer.simulationDimension = params.getSimulationDimension();

  constantsToTransfer.nx  = static_cast<unsigned int>(fullDimSizes.nx);
  constantsToTransfer.ny  = static_cast<unsigned int>(fullDimSizes.ny);
  constantsToTransfer.nz  = static_cast<unsigned int>(fullDimSizes.nz);
  constantsToTransfer.nElements = static_cast<unsigned int>(fullDimSizes.nElements());

  constantsToTransfer.nxComplex = static_cast<unsigned int>(reducedDimSizes.nx);
  constantsToTransfer.nyComplex = static_cast<unsigned int>(reducedDimSizes.ny);
  constantsToTransfer.nzComplex = static_cast<unsigned int>(reducedDimSizes.nz);
  constantsToTransfer.nElementsComplex = static_cast<unsigned int>(reducedDimSizes.nElements());

  constantsToTransfer.fftDivider  = 1.0f / fullDimSizes.nElements();
  constantsToTransfer.fftDividerX = 1.0f / fullDimSizes.nx;
  constantsToTransfer.fftDividerY = 1.0f / fullDimSizes.ny;
  constantsToTransfer.fftDividerZ = 1.0f / fullDimSizes.nz;

  constantsToTransfer.dt      = params.getDt();
  constantsToTransfer.dtBy2   = params.getDt() * 2.0f;
  constantsToTransfer.c2      = params.getC2Scalar();

  constantsToTransfer.rho0      = params.getRho0Scalar();
  constantsToTransfer.dtRho0    = params.getRho0Scalar() * params.getDt();
  constantsToTransfer.dtRho0Sgx = params.getDtRho0SgxScalar();
  constantsToTransfer.dtRho0Sgy = params.getDtRho0SgyScalar(),
  constantsToTransfer.dtRho0Sgz = params.getDtRho0SgzScalar(),

  constantsToTransfer.bOnA      = params.getBOnAScalar();
  constantsToTransfer.absorbTau = params.getAbsorbTauScalar();
  constantsToTransfer.absorbEta = params.getAbsorbEtaScalar();

  // Source masks
  constantsToTransfer.presureSourceSize = static_cast<unsigned int>(params.getPressureSourceIndexSize());
  constantsToTransfer.presureSourceMode = params.getPressureSourceMode();
  constantsToTransfer.presureSourceMany = static_cast<unsigned int>(params.getPressureSourceMany());

  constantsToTransfer.velocitySourceSize = static_cast<unsigned int>(params.getVelocitySourceIndexSize());
  constantsToTransfer.velocitySourceMode = params.getVelocitySourceMode();
  constantsToTransfer.velocitySourceMany = static_cast<unsigned int>(params.getVelocitySourceMany());

  constantsToTransfer.copyToDevice();
}// end of setUpDeviceConstants
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Check whether the CUDA driver version installed is sufficient for the code. If anything goes wrong,
 * throw an exception and exit
 */
void CudaParameters::checkCudaVersion()
{
  int cudaRuntimeVersion;
  int cudaDriverVersion;

  if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
  {
    throw std::runtime_error(kErrFmtCannotReadCudaVersion);
  }

  if (cudaDriverGetVersion(&cudaDriverVersion) != cudaSuccess)
  {
    throw std::runtime_error(kErrFmtCannotReadCudaVersion);
  }

  if (cudaDriverVersion < cudaRuntimeVersion)
  {
    throw std::runtime_error(Logger::formatMessage(kErrFmtInsufficientCudaDriver,
                                                  cudaRuntimeVersion / 1000, (cudaRuntimeVersion % 100) / 10,
                                                  cudaDriverVersion  / 1000, (cudaDriverVersion  % 100) / 10));
  }
}// end of checkCudaVersion
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check whether the GPU has SM 3.0 at least.
 */
bool CudaParameters::checkCudaCodeVersion()
{
  return (SolverCudaKernels<Parameters::SimulationDimension::k2D, true, true, true, true>::
          getCudaCodeVersion() >= 30);
}// end of checkCudaCodeVersion
//----------------------------------------------------------------------------------------------------------------------

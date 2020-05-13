/**
 * @file      SolverCudaKernels.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation containing all cuda kernels used in the GPU implementation of the k-space solver.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 March     2013, 13:10 (created) \n
 *            11 February  2020, 16:14 (revised)
 *
 * @copyright Copyright (C) 2013 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <cuComplex.h>

#include <KSpaceSolver/SolverCudaKernels.cuh>
#include <Parameters/CudaDeviceConstants.cuh>
#include <Containers/CudaMatrixContainer.cuh>

#include <Logger/Logger.h>
#include <Utils/CudaUtils.cuh>

/// Shortcut for matrix id datatype.
using MI = MatrixContainer::MatrixIdx;
/// Shortcut for Simulation dimension datatype.
using SD = Parameters::SimulationDimension;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

template class SolverCudaKernels<SD::k3D,  true,  true, true,  true>;
template class SolverCudaKernels<SD::k3D,  true,  true, true,  false>;
template class SolverCudaKernels<SD::k3D,  true,  true, false, true>;
template class SolverCudaKernels<SD::k3D,  true,  true, false, false>;

template class SolverCudaKernels<SD::k3D,  true,  false, true,  true>;
template class SolverCudaKernels<SD::k3D,  true,  false, true,  false>;
template class SolverCudaKernels<SD::k3D,  true,  false, false, true>;
template class SolverCudaKernels<SD::k3D,  true,  false, false, false>;

template class SolverCudaKernels<SD::k3D,  false,  true, true,  true>;
template class SolverCudaKernels<SD::k3D,  false,  true, true,  false>;
template class SolverCudaKernels<SD::k3D,  false,  true, false, true>;
template class SolverCudaKernels<SD::k3D,  false,  true, false, false>;

template class SolverCudaKernels<SD::k3D,  false,  false, true,  true>;
template class SolverCudaKernels<SD::k3D,  false,  false, true,  false>;
template class SolverCudaKernels<SD::k3D,  false,  false, false, true>;
template class SolverCudaKernels<SD::k3D,  false,  false, false, false>;


template class SolverCudaKernels<SD::k2D,  true,  true, true,  true>;
template class SolverCudaKernels<SD::k2D,  true,  true, true,  false>;
template class SolverCudaKernels<SD::k2D,  true,  true, false, true>;
template class SolverCudaKernels<SD::k2D,  true,  true, false, false>;

template class SolverCudaKernels<SD::k2D,  true,  false, true,  true>;
template class SolverCudaKernels<SD::k2D,  true,  false, true,  false>;
template class SolverCudaKernels<SD::k2D,  true,  false, false, true>;
template class SolverCudaKernels<SD::k2D,  true,  false, false, false>;

template class SolverCudaKernels<SD::k2D,  false,  true, true,  true>;
template class SolverCudaKernels<SD::k2D,  false,  true, true,  false>;
template class SolverCudaKernels<SD::k2D,  false,  true, false, true>;
template class SolverCudaKernels<SD::k2D,  false,  true, false, false>;

template class SolverCudaKernels<SD::k2D,  false,  false, true,  true>;
template class SolverCudaKernels<SD::k2D,  false,  false, true,  false>;
template class SolverCudaKernels<SD::k2D,  false,  false, false, true>;
template class SolverCudaKernels<SD::k2D,  false,  false, false, false>;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Variables -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @var      cudaDeviceConstants
 * @brief    This variable holds basic simulation constants for GPU.
 * @details  This variable holds necessary simulation constants in the cuda GPU memory.
 *           The variable is defined in CudaDeviceConstants.cu
 */
extern __constant__ CudaDeviceConstants cudaDeviceConstants;

/**
 * @var      cudaMatrixContainer
 * @brief    This variable holds pointer data to device matrices present in MatrixContainer.
 * @details  The variable is defined in CudaMatrixContainer.cu
 */
extern __constant__ CudaMatrixContainer<MatrixContainer::getMatrixIdxCount()> cudaMatrixContainer;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get block size for 1D kernels.
 * @return 1D block size.
 */
inline int getSolverBlockSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverBlockSize1D();
};// end of getSolverBlockSize1D
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get grid size for 1D kernels.
 * @return 1D grid size
 */
inline int getSolverGridSize1D()
{
  return Parameters::getInstance().getCudaParameters().getSolverGridSize1D();
};// end of getSolverGridSize1D
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting 2D or 3D coordinates using a templated method.
 *
 * @tparam simulationDimension - Number of dimensions.
 * @param [in] i - index.
 *
 * @return 2D/3D coordinates.
 */
template <SD simulationDimension>
inline __device__ dim3 getRealCoords(const unsigned int i)
{
  return (simulationDimension == SD::k3D) ? getReal3DCoords(i) : getReal2DCoords(i);
}// end of getRealCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting 2D or 3D coordinates using a templated method.
 *
 * @tparam simulationDimension - Number of dimensions.
 * @param [in] i - index.
 *
 * @return 2D/3D coordinates.
 */
template <SD simulationDimension>
inline __device__ dim3 getComplexCoords(const unsigned int i)
{
  return (simulationDimension == SD::k3D) ? getComplex3DCoords(i) : getComplex2DCoords(i);
}// end of getRealCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting real matrix data from the cudaMatrixContainer.

 * @param [in] matrixIdx - Matrix idx in the container.
 *
 * @return Pointer to raw data or nullptr if the matrix does not exist.
 */
inline __device__ float* getRealData(MI matrixIdx)
{
  return cudaMatrixContainer.getRealData(matrixIdx);
};// end of getRealData
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting index matrix data from the cudaMatrixContainer.
 *
 * @param [in] matrixIdx - Matrix idx in the container.
 *
 * @return Pointer to raw data or nullptr if the matrix does not exist.
 */
inline __device__ size_t* getIndexData(MI matrixIdx)
{
  return cudaMatrixContainer.getIndexData(matrixIdx);
};// end of getIndexData
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting complex matrix data from the cudaMatrixContainer.
 *
 * @param [in] matrixIdx - Matrix idx in the container.
 *
 * @return Pointer to raw data or nullptr if the matrix does not exist.
 */
inline __device__ cuFloatComplex* getComplexData(MI matrixIdx)
{
  return cudaMatrixContainer.getComplexData(matrixIdx);
};// end of getComplexData
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Kernel to find out the version of the code.
 * The list of GPUs can be found at https://en.wikipedia.org/wiki/CUDA
 *
 * @param [out] cudaCodeVersion - Version of CUDA architecture.
 */
__global__ void cudaGetCudaCodeVersion(int* cudaCodeVersion)
{
  *cudaCodeVersion = -1;

  // Read __CUDA_ARCH__ only in actual kernel compilation pass.
  // NVCC does some more passes, where it isn't defined.
  #ifdef __CUDA_ARCH__
    *cudaCodeVersion = (__CUDA_ARCH__ / 10);
  #endif
}// end of cudaGetCudaCodeVersion
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the cuda architecture the code was compiled with.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
int SolverCudaKernels<simulationDimension,
                      rho0ScalarFlag,
                      bOnAScalarFlag,
                      c0ScalarFlag,
                      alphaCoefScalarFlag>::
getCudaCodeVersion()
{
  // Host and device pointers, data copied over zero copy memory
  int* hCudaCodeVersion;
  int* dCudaCodeVersion;

  // returned value
  int cudaCodeVersion = 0;
  cudaError_t cudaError;

  // Allocate for zero copy
  cudaError = cudaHostAlloc<int>(&hCudaCodeVersion,
                                 sizeof(int),
                                 cudaHostRegisterPortable | cudaHostRegisterMapped);

  // If the device is busy, return 0 - the GPU is not supported
  if (cudaError == cudaSuccess)
  {
    cudaCheckErrors(cudaHostGetDevicePointer<int>(&dCudaCodeVersion, hCudaCodeVersion, 0));

    // Find out the cuda code version
    cudaGetCudaCodeVersion<<<1,1>>>(dCudaCodeVersion);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess)
    {
      // The GPU architecture is not supported
      cudaCodeVersion = 0;
    }
    else
    {
      cudaCodeVersion = *hCudaCodeVersion;
    }

    cudaCheckErrors(cudaFreeHost(hCudaCodeVersion));
  }

  return (cudaCodeVersion);
}// end of getCudaCodeVersion
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute spectral part of pressure gradient in between FFTs.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 * \endcode
 */
  template<SD simulationDimension>
__global__ void cudaComputePressureGradient()
{
  const cuFloatComplex* ddxKShiftPos = getComplexData(MI::kDdxKShiftPosR);
  const cuFloatComplex* ddyKShiftPos = getComplexData(MI::kDdyKShiftPos);
  const cuFloatComplex* ddzKShiftPos = getComplexData(MI::kDdzKShiftPos);

  const float*  kappa = getRealData(MI::kKappa);

  cuFloatComplex* ifftX = getComplexData(MI::kTempCufftX);
  cuFloatComplex* ifftY = getComplexData(MI::kTempCufftY);
  cuFloatComplex* ifftZ = getComplexData(MI::kTempCufftZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplexCoords<simulationDimension>(i);

    const cuFloatComplex eKappa = ifftX[i] * kappa[i];

    ifftX[i] = cuCmulf(eKappa, ddxKShiftPos[coords.x]);
    ifftY[i] = cuCmulf(eKappa, ddyKShiftPos[coords.y]);
    if (simulationDimension == SD::k3D)
    {
      ifftZ[i] = cuCmulf(eKappa, ddzKShiftPos[coords.z]);
    }
  }
}// end of cudaComputePressureGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 *  Interface to kernel to compute spectral part of pressure gradient in between FFTs.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureGradient()
{
  cudaComputePressureGradient<simulationDimension>
                             <<<getSolverGridSize1D(),getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic velocity for a uniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 *<b> Matlab code: </b> \code
 *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX);
 *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY);
 *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ);
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag>
__global__ void cudaComputeVelocityUniform()
{
  const float* dtRho0SgxMatrix = getRealData(MI::kDtRho0Sgx);
  const float* dtRho0SgyMatrix = getRealData(MI::kDtRho0Sgy);
  const float* dtRho0SgzMatrix = getRealData(MI::kDtRho0Sgz);

  const float* ifftX = getRealData(MI::kTemp1RealND);
  const float* ifftY = getRealData(MI::kTemp2RealND);
  const float* ifftZ = getRealData(MI::kTemp3RealND);

  const float* pmlX  = getRealData(MI::kPmlXSgx);
  const float* pmlY  = getRealData(MI::kPmlYSgy);
  const float* pmlZ  = getRealData(MI::kPmlZSgz);

  float* uxSgx = getRealData(MI::kUxSgx);
  float* uySgy = getRealData(MI::kUySgy);
  float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    const float dtRho0Sgx  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgx : dtRho0SgxMatrix[i];
    const float dtRho0Sgy  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgy : dtRho0SgyMatrix[i];

    uxSgx[i] = (uxSgx[i] * pmlX[coords.x] - cudaDeviceConstants.fftDivider * ifftX[i] * dtRho0Sgx) * pmlX[coords.x];
    uySgy[i] = (uySgy[i] * pmlY[coords.y] - cudaDeviceConstants.fftDivider * ifftY[i] * dtRho0Sgy) * pmlY[coords.y];

    if (simulationDimension == SD::k3D)
    {
      const float dtRho0Sgz = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgz : dtRho0SgzMatrix[i];

      uzSgz[i] = (uzSgz[i] * pmlZ[coords.z] - cudaDeviceConstants.fftDivider * ifftZ[i] * dtRho0Sgz) * pmlZ[coords.z];
    }
  }
}// end of cudaComputeVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to the cuda kernel computing new values for particle velocity on a uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityUniform()
{
  cudaComputeVelocityUniform<simulationDimension, rho0ScalarFlag>
                            <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to Compute acoustic velocity for homogenous medium and nonuniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx)  ...
 *                  - dt .* rho0_sgx_inv .* dxudxnSgx.* real(ifftX));
 *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) ...
 *                  - dt .* rho0_sgy_inv .* dyudynSgy.* real(ifftY);
 *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz)
 *                  - dt .* rho0_sgz_inv .* dzudznSgz.* real(ifftZ);
 *\endcode
 */
template<SD simulationDimension>
__global__ void cudaComputeVelocityHomogeneousNonuniform()
{
  const float dividerX = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float dividerY = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;;
  const float dividerZ = (simulationDimension == SD::k3D)
                            ? cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider : 1.0f;

  const float* dxudxnSgx = getRealData(MI::kDxudxnSgx);
  const float* dyudynSgy = getRealData(MI::kDyudynSgy);
  const float* dzudznSgz = getRealData(MI::kDzudznSgz);

  const float* ifftX = getRealData(MI::kTemp1RealND);
  const float* ifftY = getRealData(MI::kTemp2RealND);
  const float* ifftZ = getRealData(MI::kTemp3RealND);

  const float* pmlX  = getRealData(MI::kPmlXSgx);
  const float* pmlY  = getRealData(MI::kPmlYSgy);
  const float* pmlZ = getRealData(MI::kPmlZSgz);

  float* uxSgx = getRealData(MI::kUxSgx);
  float* uySgy = getRealData(MI::kUySgy);
  float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    uxSgx[i] = (uxSgx[i] * pmlX[coords.x] - dividerX * dxudxnSgx[coords.x] * ifftX[i]) * pmlX[coords.x];
    uySgy[i] = (uySgy[i] * pmlY[coords.y] - dividerY * dyudynSgy[coords.y] * ifftY[i]) * pmlY[coords.y];

    if (simulationDimension == SD::k3D)
    {
      uzSgz[i] = (uzSgz[i] * pmlZ[coords.z] - dividerZ * dzudznSgz[coords.z] * ifftZ[i]) * pmlZ[coords.z];
    }
  }// for
}// end of cudaComputeVelocityHomogeneouosNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to compute acoustic velocity for homogenous medium and nonuniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityHomogeneousNonuniform()
{
  cudaComputeVelocityHomogeneousNonuniform<simulationDimension>
                                          <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in the x direction.
 */
__global__ void cudaComputeVelocityShiftInX()
{
  const cuFloatComplex* xShiftNegR = getComplexData(MI::kXShiftNegR);

  cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const auto  x = i % cudaDeviceConstants.nxComplex;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], xShiftNegR[x]) * cudaDeviceConstants.fftDividerX;
  }
}// end of cudaComputeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the x axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityShiftInX()
{
  cudaComputeVelocityShiftInX<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
 }// end of computeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in Y. The matrix is XY transposed.
 */
__global__ void cudaComputeVelocityShiftInY()
{
  const auto nyR       = cudaDeviceConstants.ny / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * nyR * cudaDeviceConstants.nz;

  const cuFloatComplex* yShiftNegR = getComplexData(MI::kYShiftNegR);

  cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // Rotated dimensions
    const auto  y = i % nyR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], yShiftNegR[y]) * cudaDeviceConstants.fftDividerY;
  }
}// end of cudaComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the y axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                          rho0ScalarFlag,
                          bOnAScalarFlag,
                          c0ScalarFlag,
                          alphaCoefScalarFlag>::
computeVelocityShiftInY()
{
  cudaComputeVelocityShiftInY<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of ComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in Z. The matrix is XZ transposed.
 *
 */
__global__ void cudaComputeVelocityShiftInZ()
{
  const auto nzR       = cudaDeviceConstants.nz / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * cudaDeviceConstants.ny * nzR;

  const cuFloatComplex* zShiftNegR = getComplexData(MI::kZShiftNegR);

  cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // Rotated dimensions
    const auto  z = i % nzR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], zShiftNegR[z]) * cudaDeviceConstants.fftDividerZ;
  }
}// end of cudaComputeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the z axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityShiftInZ()
{
  cudaComputeVelocityShiftInZ<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Kernel to compute spatial part of the velocity gradient in between FFTs on uniform grid.
 * Complex numbers are passed as float2 structures.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx));
 *  bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy));
 *  bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz));
 * \endcode
 */
template<SD simulationDimension>
__global__  void cudaComputeVelocityGradient()
{
  const cuFloatComplex* ddxKShiftNeg = getComplexData(MI::kDdxKShiftNegR);
  const cuFloatComplex* ddyKShiftNeg = getComplexData(MI::kDdyKShiftNeg);
  const cuFloatComplex* ddzKShiftNeg = getComplexData(MI::kDdzKShiftNeg);

  const float* kappa   = getRealData(MI::kKappa);

  cuFloatComplex* fftX = getComplexData(MI::kTempCufftX);
  cuFloatComplex* fftY = getComplexData(MI::kTempCufftY);
  cuFloatComplex* fftZ = getComplexData(MI::kTempCufftZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplexCoords<simulationDimension>(i);

    const float eKappa = kappa[i] * cudaDeviceConstants.fftDivider;

    fftX[i] = cuCmulf(fftX[i] * eKappa, ddxKShiftNeg[coords.x]);
    fftY[i] = cuCmulf(fftY[i] * eKappa, ddyKShiftNeg[coords.y]);

    if (simulationDimension == SD::k3D)
    {
      fftZ[i] = cuCmulf(fftZ[i] * eKappa, ddzKShiftNeg[coords.z]);
    }
  }// for
}// end of cudaComputeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute spatial part of the velocity gradient in between FFTs on uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityGradient()
{
  cudaComputeVelocityGradient<simulationDimension>
                             <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to shift gradient of acoustic velocity on non-uniform grid.
 * @tparam simulationDimension - Dimensionality of the simulation.
 */
template<SD simulationDimension>
__global__  void cudaComputeVelocityGradientShiftNonuniform()
{
  const float* duxdxn = getRealData(MI::kDxudxn);
  const float* duydyn = getRealData(MI::kDyudyn);
  const float* duzdzn = getRealData(MI::kDzudzn);

  float* duxdx = getRealData(MI::kDuxdx);
  float* duydy = getRealData(MI::kDuydy);
  float* duzdz = getRealData(MI::kDuzdz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    duxdx[i] *= duxdxn[coords.x];
    duydy[i] *= duydyn[coords.y];

    if (simulationDimension == SD::k3D)
    {
      duzdz[i] *= duzdzn[coords.z];
    }
  }
}// end of cudaComputeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to shift gradient of acoustic velocity on non-uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityGradientShiftNonuniform()
{
  cudaComputeVelocityGradientShiftNonuniform<simulationDimension>
                                            <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic density for non-linear case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
 *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
 *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
 *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag>
__global__ void cudaComputeDensityNonlinear()
{
  const float* pmlX       = getRealData(MI::kPmlX);
  const float* pmlY       = getRealData(MI::kPmlY);
  const float* pmlZ       = getRealData(MI::kPmlZ);

  const float* duxdx      = getRealData(MI::kDuxdx);
  const float* duydy      = getRealData(MI::kDuydy);
  const float* duzdz      = getRealData(MI::kDuzdz);

  const float* rho0Matrix = getRealData(MI::kRho0);

  float* rhoX = getRealData(MI::kRhoX);
  float* rhoY = getRealData(MI::kRhoY);
  float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    const float rho0      = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    // 3D and 2D summation
    const float sumRhos   = (simulationDimension == SD::k3D) ? (rhoX[i] + rhoY[i] + rhoZ[i])
                                                             : (rhoX[i] + rhoY[i]);

    const float sumRhosDt = (2.0f * sumRhos + rho0) * cudaDeviceConstants.dt;

    rhoX[i] = pmlX[coords.x] * ((pmlX[coords.x] * rhoX[i]) - sumRhosDt * duxdx[i]);
    rhoY[i] = pmlY[coords.y] * ((pmlY[coords.y] * rhoY[i]) - sumRhosDt * duydy[i]);

    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = pmlZ[coords.z] * ((pmlZ[coords.z] * rhoZ[i]) - sumRhosDt * duzdz[i]);
    }
  }
}//end of cudaComputeDensityNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute acoustic density for non-linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeDensityNonlinear()
{
  cudaComputeDensityNonlinear<simulationDimension, rho0ScalarFlag>
                             <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeDensityNonlinear
//-----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic density for linear case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
 *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
 *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag>
__global__ void cudaComputeDensityLinear()
{
  const float* pmlX       = getRealData(MI::kPmlX);
  const float* pmlY       = getRealData(MI::kPmlY);
  const float* pmlZ       = getRealData(MI::kPmlZ);

  const float* duxdx      = getRealData(MI::kDuxdx);
  const float* duydy      = getRealData(MI::kDuydy);
  const float* duzdz      = getRealData(MI::kDuzdz);

  const float* rho0Matrix = getRealData(MI::kRho0);

  float* rhoX = getRealData(MI::kRhoX);
  float* rhoY = getRealData(MI::kRhoY);
  float* rhoZ = getRealData(MI::kRhoZ);


  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    const float dtRho0  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0 : cudaDeviceConstants.dt * rho0Matrix[i];

    rhoX[i] = pmlX[coords.x] * (pmlX[coords.x] * rhoX[i] - dtRho0 * duxdx[i]);
    rhoY[i] = pmlY[coords.y] * (pmlY[coords.y] * rhoY[i] - dtRho0 * duydy[i]);

    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = pmlZ[coords.z] * (pmlZ[coords.z] * rhoZ[i] - dtRho0 * duzdz[i]);
    }
  }
}// end of cudaComputeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute acoustic density for linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeDensityLinear()
{
  cudaComputeDensityLinear<simulationDimension, rho0ScalarFlag>
                          <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sums sub-terms for new pressure in non-linear lossless case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam isRho0Scalar        - Is density homogeneous?
 * @tparam isBOnAScalar        - Is nonlinearity homogeneous?
 * @tparam isC2Scalar          - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear adiabatic equation of state
 *  p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag>
__global__ void cudaSumPressureNonlinearLossless()
{
  const float* rhoX = getRealData(MI::kRhoX);
  const float* rhoY = getRealData(MI::kRhoY);
  const float* rhoZ = getRealData(MI::kRhoZ);

  const float* rho0Matrix = getRealData(MI::kRho0);
  const float* bOnAMatrix = getRealData(MI::kBOnA);
  const float* c2Matrix   = getRealData(MI::kC2);

  float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    const float bOnA = (bOnAScalarFlag) ? cudaDeviceConstants.bOnA : bOnAMatrix[i];
    const float c2   = (c0ScalarFlag)   ? cudaDeviceConstants.c2   : c2Matrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                          : rhoX[i] + rhoY[i];

    p[i] = c2 * (rhoSum + (bOnA * (rhoSum * rhoSum) / (2.0f * rho0)));
  }
}// end of cudaSumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms for new pressure in non-linear lossless case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureNonlinearLossless()
{
  cudaSumPressureNonlinearLossless<simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag>
                                  <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute three temporary sums in the new pressure formula in non-linear power law case.
 *
 * @tparam simulationDimension      - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag           - Is rho0 a scalar value (homogeneous)?
 * @tparam bOnAScalarFlag           - Is B on A homogeneous?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
 * @param [out] nonlinearTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz);
 * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag>
__global__ void cudaComputePressureTermsNonlinearPowerLaw(float* densitySum,
                                                          float* nonlinearTerm,
                                                          float* velocityGradientSum)
{
  const float* rhoX       = getRealData(MI::kRhoX);
  const float* rhoY       = getRealData(MI::kRhoY);
  const float* rhoZ       = getRealData(MI::kRhoZ);

  const float* duxdx      = getRealData(MI::kDuxdx);
  const float* duydy      = getRealData(MI::kDuydy);
  const float* duzdz      = getRealData(MI::kDuzdz);

  const float* rho0Matrix = getRealData(MI::kRho0);
  const float* bOnAMatrix = getRealData(MI::kBOnA);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    const float bonA = (bOnAScalarFlag) ? cudaDeviceConstants.bOnA : bOnAMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? (rhoX[i] + rhoY[i] + rhoZ[i])
                                                          : (rhoX[i] + rhoY[i]);

    const float duSum  = (simulationDimension == SD::k3D) ? (duxdx[i] + duydy[i] + duzdz[i])
                                                          : (duxdx[i] + duydy[i]);

    densitySum[i]          = rhoSum;
    nonlinearTerm[i]       = ((bonA * rhoSum * rhoSum) / (2.0f * rho0)) + rhoSum;
    velocityGradientSum[i] = rho0 * duSum;
  }
}// end of cudaComputePressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute three temporary sums in the new pressure formula in non-linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureTermsNonlinearPowerLaw(RealMatrix& densitySum,
                                      RealMatrix& nonlinearTerm,
                                      RealMatrix& velocityGradientSum)
{
  cudaComputePressureTermsNonlinearPowerLaw<simulationDimension, rho0ScalarFlag, bOnAScalarFlag>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                   (densitySum.getDeviceData(),
                                    nonlinearTerm.getDeviceData(),
                                    velocityGradientSum.getDeviceData());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute absorbing term with abosrbNabla1 and  absorbNabla2.
 *
 * <b>Matlab code:</b> \code
 *  fftPart1 = absorbNabla1 .* fftPart1;
 *  fftPart2 = absorbNabla2 .* fftPart2;
 * \endcode
 */
__global__ void cudaComputeAbsorbtionTerm()
{

  const float* absorbNabla1 = getRealData(MI::kAbsorbNabla1);
  const float* absorbNabla2 = getRealData(MI::kAbsorbNabla2);

  cuFloatComplex* fftPart1  = getComplexData(MI::kTempCufftX);
  cuFloatComplex* fftPart2  = getComplexData(MI::kTempCufftY);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    fftPart1[i] *= absorbNabla1[i];
    fftPart2[i] *= absorbNabla2[i];
  }
}// end of computeAbsorbtionTerm
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to computes absorbing term with abosrbNabla1 and  absorbNabla2.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeAbsorbtionTerm()
{
  cudaComputeAbsorbtionTerm<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeAbsorbtionTerm
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in non-linear power law case.
 *
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?

 * @param [in] nonlinearTerm   - Nonlinear term
 * @param [in] absorbTauTerm   - Absorb tau term from the pressure eq.
 * @param [in] absorbEtaTerm   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz);
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear absorbing equation of state
 *  p = c0.^2 .* (...
 *                nonlinearTerm ...
 *                + absorb_tau .* absorbTauTerm...
 *                - absorb_eta .* absorbEtaTerm...
 *                );
 * \endcode
 */
template<bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
__global__ void cudaSumPressureTermsNonlinearPowerlaw(const float* nonlinearTerm,
                                                      const float* absorbTauTerm,
                                                      const float* absorbEtaTerm)
{
  const float* c2Matrix        = getRealData(MI::kC2);
  const float* absorbTauMatrix = getRealData(MI::kAbsorbTau);
  const float* absorbEtaMatrix = getRealData(MI::kAbsorbEta);

  float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];
    const float absorbEta = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbEta : absorbEtaMatrix[i];

    p[i] = c2 * (nonlinearTerm[i] + (cudaDeviceConstants.fftDivider *
                                     ((absorbTauTerm[i] * absorbTau) - (absorbEtaTerm[i] * absorbEta))));
  }
}// end of cudaSumPressureTermsNonlinearPowerlaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel Sum sub-terms to compute new pressure in non-linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureTermsNonlinearPowerLaw(const RealMatrix& nonlinearTerm,
                                  const RealMatrix& absorbTauTerm,
                                  const RealMatrix& absorbEtaTerm)
{
  cudaSumPressureTermsNonlinearPowerlaw<c0ScalarFlag, alphaCoefScalarFlag>
                                       <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                       (nonlinearTerm.getDeviceData(),
                                        absorbTauTerm.getDeviceData(),
                                        absorbEtaTerm.getDeviceData());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sum sub-terms to compute new pressure in  nonlinear stokes case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam isC2Scalar          - Is sound speed homogeneous?
 * @tparam isBOnAScalar        - Is nonlinearity homogeneous?
 * @tparam isRho0Scalar        - Is density homogeneous?
 * @tparam isAbsorbTauScalar   - Is absorption homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  p = c0.^2 .* ( ...
 *      (rhox + rhoy + rhoz) ...
 *       + absorb_tau .* rho0 .* (duxdx + duydy) ...
 *       + medium.BonA .* (rhox + rhoy).^2 ./ (2 .* rho0));
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
__global__ void cudaSumPressureNonlinearStokes()
{
  const float* rhoX  = getRealData(MI::kRhoX);
  const float* rhoY  = getRealData(MI::kRhoY);
  const float* rhoZ  = getRealData(MI::kRhoZ);

  const float* duxdx = getRealData(MI::kDuxdx);
  const float* duydy = getRealData(MI::kDuydy);
  const float* duzdz = getRealData(MI::kDuzdz);

  const float* rho0Matrix      = getRealData(MI::kRho0);
  const float* bOnAMatrix      = getRealData(MI::kBOnA);
  const float* c2Matrix        = getRealData(MI::kC2);
  const float* absorbTauMatrix = getRealData(MI::kAbsorbTau);

  float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0      = (rho0ScalarFlag)      ? cudaDeviceConstants.rho0      : rho0Matrix[i];
    const float bOnA      = (bOnAScalarFlag)      ? cudaDeviceConstants.bOnA      : bOnAMatrix[i];
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i]  + rhoY[i]  + rhoZ[i]  : rhoX[i]  + rhoY[i];
    const float duSum  = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i] : duxdx[i] + duydy[i];

    p[i] = c2 * (rhoSum + absorbTau * rho0 * duSum + ((bOnA * rhoSum * rhoSum) / (2.0f * rho0)));
  }
}// end of cudaSumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in  nonlinear stokes case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureNonlinearStokes()
{
  cudaSumPressureNonlinearStokes<simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag, alphaCoefScalarFlag>
                                <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Cuda kernel to sum sub-terms for new pressure in linear lossless case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam c0ScalarFlag        - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a linear adiabatic equation of state
 *  p = c0.^2 .* (rhox + rhoy + rhoz);
 * \endcode
 */
template<SD   simulationDimension,
         bool c0ScalarFlag>
__global__ void cudaSumPressureLinearLossless()
{
  const float* c2Matrix  = getRealData(MI::kC2);

  const float* rhoX = getRealData(MI::kRhoX);
  const float* rhoY = getRealData(MI::kRhoY);
  const float* rhoZ = getRealData(MI::kRhoZ);

  float* p  = getRealData(MI::kP);

  for (auto  i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2         = (c0ScalarFlag) ? cudaDeviceConstants.c2 : c2Matrix[i];

    const float sumDensity = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                              : rhoX[i] + rhoY[i];
    p[i] = c2 * sumDensity;
  }
}// end of cudaSumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms for new pressure in linear lossless case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                          rho0ScalarFlag,
                          bOnAScalarFlag,
                          c0ScalarFlag,
                          alphaCoefScalarFlag>::
sumPressureLinearLossless()
{
  cudaSumPressureLinearLossless<simulationDimension, c0ScalarFlag>
                               <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute two temporary sums in the new pressure formula for linear power law case.
 *
 * @tparam simulationDimension      - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag           - Is density homogeneous?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
 * @param [out] velocityGradientSum - rho0 * (duxdx + duydy + duzdz);
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag>
__global__ void cudaComputePressureTermsLinearPowerLaw(float* densitySum,
                                                       float* velocityGradientSum)
{
  const float* rhoX = getRealData(MI::kRhoX);
  const float* rhoY = getRealData(MI::kRhoY);
  const float* rhoZ = getRealData(MI::kRhoZ);

  const float* duxdx = getRealData(MI::kDuxdx);
  const float* duydy = getRealData(MI::kDuydy);
  const float* duzdz = getRealData(MI::kDuzdz);

  const float* rho0Matrix = getRealData(MI::kRho0);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0  = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];

    densitySum[i]     = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                         : rhoX[i] + rhoY[i];
    const float duSum = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i]
                                                         : duxdx[i] + duydy[i];

    velocityGradientSum[i] = rho0 * duSum;
  }
}// end of cudaComputePressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute two temporary sums in the new pressure formula for linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureTermsLinearPowerLaw(RealMatrix& densitySum,
                                   RealMatrix& velocityGradientSum)
{
  cudaComputePressureTermsLinearPowerLaw<simulationDimension, rho0ScalarFlag>
                                        <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                        (densitySum.getDeviceData(),
                                         velocityGradientSum.getDeviceData());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computePressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in linear power law case.
 *
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
 *
 * @param [in] absorbTauTerm   - Absorb tau term from the pressure eq.
 * @param [in] absorbEtaTerm   - Absorb tau term from the pressure eq.
 * @param [in] densitySum      - Sum of acoustic density.
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear absorbing equation of state
 *  p = c0.^2 .* (...
 *                densitySum
 *                + absorb_tau .* absorbTauTerm...
 *                - absorb_eta .* absorbEtaTerm...
 *                );
 * \endcode
 */
template<bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
__global__ void cudaSumPressureTermsLinearPowerLaw(const float* absorbTauTerm,
                                                   const float* absorbEtaTerm,
                                                   const float* densitySum)
{
  const float* c2Matrix        = getRealData(MI::kC2);
  const float* absorbTauMatrix = getRealData(MI::kAbsorbTau);
  const float* absorbEtaMatrix = getRealData(MI::kAbsorbEta);

  float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];
    const float absorbEta = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbEta : absorbEtaMatrix[i];

    p[i] = c2 * (densitySum[i] + (cudaDeviceConstants.fftDivider *
                (absorbTauTerm[i] * absorbTau - absorbEtaTerm[i] * absorbEta)));
  }
}// end of cudaSumPressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureTermsLinearPowerLaw(const RealMatrix& absorbTauTerm,
                               const RealMatrix& absorbEtaTerm,
                               const RealMatrix& densitySum)
{
  cudaSumPressureTermsLinearPowerLaw<c0ScalarFlag, alphaCoefScalarFlag>
                                    <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                                    (absorbTauTerm.getDeviceData(),
                                     absorbEtaTerm.getDeviceData(),
                                     densitySum.getDeviceData());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in linear stokes case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
 * \endcode
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
__global__ void cudaSumPressureLinearStokes()
{
  const float* rhoX  = getRealData(MI::kRhoX);
  const float* rhoY  = getRealData(MI::kRhoY);
  const float* rhoZ  = getRealData(MI::kRhoZ);

  const float* duxdx = getRealData(MI::kDuxdx);
  const float* duydy = getRealData(MI::kDuydy);
  const float* duzdz = getRealData(MI::kDuzdz);

  const float* rho0Matrix      = getRealData(MI::kRho0);
  const float* c2Matrix        = getRealData(MI::kC2);
  const float* absorbTauMatrix = getRealData(MI::kAbsorbTau);

  float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0      = (rho0ScalarFlag)      ? cudaDeviceConstants.rho0      : rho0Matrix[i];
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i]  + rhoY[i]  + rhoZ[i]  : rhoX[i]  + rhoY[i];
    const float duSum  = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i] : duxdx[i] + duydy[i];

     p[i] = c2 * (rhoSum + absorbTau * rho0 * duSum);
  }
}// end of cudaSumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in linear stokes case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureLinearStokes()
{
  cudaSumPressureLinearStokes<simulationDimension, rho0ScalarFlag, c0ScalarFlag, alphaCoefScalarFlag>
                             <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sumPressureTermsLinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add pressure source to acoustic density.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @param [in] timeIndex       - Actual time step.
 */
template<SD simulationDimension>
__global__ void cudaAddPressureSource(const size_t timeIndex)
{
  const float*  pressureSourceInput = getRealData(MI::kPressureSourceInput);
  const size_t* pressureSourceIndex = getIndexData(MI::kPressureSourceIndex);

  float* rhoX = getRealData(MI::kRhoX);
  float* rhoY = getRealData(MI::kRhoY);
  float* rhoZ = getRealData(MI::kRhoZ);

  // Set 1D or 2D step for source
  const auto index2D = (cudaDeviceConstants.presureSourceMany == 0)
                          ? timeIndex : timeIndex * cudaDeviceConstants.presureSourceSize;

  // Different pressure sources
  switch (cudaDeviceConstants.presureSourceMode)
  {
    case Parameters::SourceMode::kDirichlet:
    {
      if (cudaDeviceConstants.presureSourceMany == 0)
      { // Single signal
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          }
        }
      }
      else
      { // Multiple signals
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          }
        }
      }
      break;
    }// Dirichlet

    case Parameters::SourceMode::kAdditiveNoCorrection:
    {
      if (cudaDeviceConstants.presureSourceMany == 0)
      { // Single signal
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          }
        }
      }
      else
      { // Multiple signals
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          }
        }
      }
      break;
    }
    default:
    {
      break;
    }
  }// end switch
}// end of cudaAddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add in pressure source (to acoustic density).
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addPressureSource(const MatrixContainer& container)
{
  const int sourceSize = int(container.getMatrix<IndexMatrix>(MI::kPressureSourceIndex).size());

  // Grid size is calculated based on the source size
  const int gridSize  = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                           ? (sourceSize  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                           :  getSolverGridSize1D();

  cudaAddPressureSource<simulationDimension>
                       <<<gridSize,getSolverBlockSize1D()>>>
                       (Parameters::getInstance().getTimeIndex());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add transducer data source to velocity x component.
 *
 * @param [in] timeIndex - Actual time step.
 */
__global__ void cudaAddTransducerSource(const size_t timeIndex)
{
  const size_t* velocitySourceIndex   = getIndexData(MI::kVelocitySourceIndex);
  const size_t* delayMask             = getIndexData(MI::kDelayMask);
  const float*  transducerSourceInput = getRealData(MI::kTransducerSourceInput);

  float* uxSgx = getRealData(MI::kUxSgx);

  for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
  {
    uxSgx[velocitySourceIndex[i]] += transducerSourceInput[delayMask[i] + timeIndex];
  }
}// end of cudaAddTransducerSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add transducer data source to velocity x component.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addTransducerSource(const MatrixContainer& container)
{
  // Cuda only supports 32bits anyway
  const int sourceSize = int(container.getMatrix<IndexMatrix>(MI::kVelocitySourceIndex).size());

  // Grid size is calculated based on the source size
  const int gridSize  = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                           ? (sourceSize  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                           : getSolverGridSize1D();

  cudaAddTransducerSource<<<gridSize, getSolverBlockSize1D()>>>
                         (Parameters::getInstance().getTimeIndex());
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddTransducerSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add in velocity source terms.
 *
 * @param [in, out] velocity            - velocity matrix to update.
 * @param [in]      velocitySourceInput - Source input to add.
 * @param [in]      velocitySourceIndex - Index matrix.
 * @param [in]      timeIndex           - Actual time step.
 */
__global__ void cudaAddVelocitySource(float*        velocity,
                                      const float*  velocitySourceInput,
                                      const size_t* velocitySourceIndex,
                                      const size_t  timeIndex)
{
  // Set 1D or 2D step for source
  const auto index2D = (cudaDeviceConstants.velocitySourceMany == 0)
                          ? timeIndex : timeIndex * cudaDeviceConstants.velocitySourceSize;

  if (cudaDeviceConstants.velocitySourceMode == Parameters::SourceMode::kDirichlet)
  {
    for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] = (cudaDeviceConstants.velocitySourceMany == 0)
                                            ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }// for
  }// end of Dirichlet

  if (cudaDeviceConstants.velocitySourceMode == Parameters::SourceMode::kAdditiveNoCorrection)
  {
    for (auto i  = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] += (cudaDeviceConstants.velocitySourceMany == 0)
                                             ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }
  }// end additive
}// end of cudaAddVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to add in velocity source terms.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addVelocitySource(RealMatrix&        velocity,
                  const RealMatrix&  velocitySourceInput,
                  const IndexMatrix& velocitySourceIndex)
{
  const int sourceSize = static_cast<int>(velocitySourceIndex.size());

  // Grid size is calculated based on the source size.
  // For small sources, a custom number of thread blocks is created,
  // otherwise, a standard number is used

  const int gridSize = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                          ? (sourceSize  + getSolverBlockSize1D() - 1 ) / getSolverBlockSize1D()
                          :  getSolverGridSize1D();

  cudaAddVelocitySource<<< gridSize, getSolverBlockSize1D()>>>
                       (velocity.getDeviceData(),
                        velocitySourceInput.getDeviceData(),
                        velocitySourceIndex.getDeviceData(),
                        Parameters::getInstance().getTimeIndex());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add scaled pressure source to acoustic density.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @param  [in] scaledSource   - Scaled source.
 */
template<SD simulationDimension>
__global__ void cudaAddPressureScaledSource(const float* scaledSource)
{
  float* rhoX = getRealData(MI::kRhoX);
  float* rhoY = getRealData(MI::kRhoY);
  float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float eScaledSource = scaledSource[i];
    rhoX[i] += eScaledSource;
    rhoY[i] += eScaledSource;
    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] += eScaledSource;
    }
  }
}// end of cudaAddPressureScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add scaled pressure source to acoustic density.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addPressureScaledSource(const RealMatrix& scaledSource)
{
  cudaAddPressureScaledSource<simulationDimension>
                             <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (scaledSource.getDeviceData());

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddPressureScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add scaled pressure source to acoustic density.
 *
 * @param [in, out] velocity     - Velocity matrix to update.
 * @param [in]      scaledSource - Scaled source.
 */
__global__ void cudaAddVelocityScaledSource(float*       velocity,
                                            const float* scaledSource)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    velocity[i] += scaledSource[i];
  }
}// end of cudaAddVelocityScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add scaled pressure source to acoustic density.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addVelocityScaledSource(RealMatrix&       velocity,
                        const RealMatrix& scaledSource)
{
  cudaAddVelocityScaledSource<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                             (velocity.getDeviceData(),
                              scaledSource.getDeviceData());
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of AddVelocityScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add pressure source to acoustic density.
 *
 * @tparam isMany            - Is the many flag set
 *
 * @param [out] scaledSource - Temporary matrix to insert source before scaling.
 * @param [in]  sourceInput  - Source input signal.
 * @param [in]  sourceIndex  - Source geometry.
 * @param [in]  sourceSize   - Size of the source.
 * @param [in]  timeIndex    - Actual time step.
 */
template<bool isMany>
__global__ void cudaInsertSourceIntoScalingMatrix(float*        scaledSource,
                                                  const float*  sourceInput,
                                                  const size_t* sourceIndex,
                                                  const size_t  sourceSize,
                                                  const size_t  timeIndex)
{
  // Set 1D or 2D step for source
  const auto index2D = (isMany) ? timeIndex * sourceSize : timeIndex;

  // Different pressure sources
  if (isMany)
  { // Multiple signals
    for (auto i = getIndex(); i < sourceSize; i += getStride())
    {
      scaledSource[sourceIndex[i]] = sourceInput[index2D + i];
    }
  }
  else
  { // Single signal
    for (auto i = getIndex(); i < sourceSize; i += getStride())
    {
      scaledSource[sourceIndex[i]] = sourceInput[index2D];
    }
  }
}// end of cudaInsertSourceIntoScalingMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to insert source signal into scaling matrix.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
insertSourceIntoScalingMatrix(RealMatrix&        scaledSource,
                              const RealMatrix&  sourceInput,
                              const IndexMatrix& sourceIndex,
                              const size_t       manyFlag)
{
  const int sourceSize = static_cast<int>(sourceIndex.size());
  // Grid size is calculated based on the source size
  const int gridSize  = (sourceSize < (getSolverGridSize1D() *  getSolverBlockSize1D()))
                          ? (sourceSize + getSolverBlockSize1D() - 1) / getSolverBlockSize1D()
                          :  getSolverGridSize1D();


  if (manyFlag == 0)
  { // Multiple signals
    cudaInsertSourceIntoScalingMatrix<false>
                                     <<<gridSize,getSolverBlockSize1D()>>>
                                     (scaledSource.getDeviceData(),
                                      sourceInput.getDeviceData(),
                                      sourceIndex.getDeviceData(),
                                      sourceIndex.size(),
                                      Parameters::getInstance().getTimeIndex());
  }
  else
  { // Single signal
    cudaInsertSourceIntoScalingMatrix<true>
                                     <<<gridSize,getSolverBlockSize1D()>>>
                                     (scaledSource.getDeviceData(),
                                      sourceInput.getDeviceData(),
                                      sourceIndex.getDeviceData(),
                                      sourceIndex.size(),
                                      Parameters::getInstance().getTimeIndex());
  }

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of insertSourceIntoScalingMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute source gradient.
 *
 * @param [in, out] sourceSpectrum  - Source spectrum.
 * @param [in]      sourceKappa     - Source kappa.
 */
__global__ void cudaComputeSourceGradient(cuFloatComplex* sourceSpectrum,
                                          const float*    sourceKappa)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    sourceSpectrum[i] *= sourceKappa[i] * cudaDeviceConstants.fftDivider;
  }
}// end of cudaComputeSourceGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute source gradient.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeSourceGradient(CufftComplexMatrix& sourceSpectrum,
                      const RealMatrix&   sourceKappa)
{
  cudaComputeSourceGradient<<<getSolverGridSize1D(), getSolverBlockSize1D()>>>
                           (sourceSpectrum.getComplexDeviceData(),
                            sourceKappa.getDeviceData());
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeSourceGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam c0ScalarFlag        - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % add the initial pressure to rho as a mass source (3D code)
 *  p = source.p0;
 *  rhox = source.p0 ./ (3 .* c.^2);
 *  rhoy = source.p0 ./ (3 .* c.^2);
 *  rhoz = source.p0 ./ (3 .* c.^2);
 * \endcode
 */
template<SD   simulationDimension,
         bool c0ScalarFlag>
__global__ void cudaAddInitialPressureSource()
{
  constexpr float dimScalingFactor = (simulationDimension == SD::k3D) ? 3.0f : 2.0f;

  const float* sourceInput = getRealData(MI::kInitialPressureSourceInput);
  const float* c2          = getRealData(MI::kC2);

  float* p    = getRealData(MI::kP);
  float* rhoX = getRealData(MI::kRhoX);
  float* rhoY = getRealData(MI::kRhoY);
  float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    p[i] = sourceInput[i];

    const float tmp = (c0ScalarFlag) ? sourceInput[i] / (dimScalingFactor * cudaDeviceConstants.c2)
                                     : sourceInput[i] / (dimScalingFactor * c2[i]);

    rhoX[i] = tmp;
    rhoY[i] = tmp;
    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = tmp;
    }
  }
}// end of cudaAddInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface for kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addInitialPressureSource()
{
  cudaAddInitialPressureSource<simulationDimension, c0ScalarFlag>
                              <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel compute acoustic velocity for initial pressure problem.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Homogenous or heterogenous medium.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = dt ./ rho0_sgx .* ifft(ux_sgx);
 *  uy_sgy = dt ./ rho0_sgy .* ifft(uy_sgy);
 *  uz_sgz = dt ./ rho0_sgz .* ifft(uz_sgz);
 * \endcode
 */
template<SD simulationDimension,
         bool rho0ScalarFlag>
__global__  void cudaComputeInitialVelocityUniform()
{
  float* uxSgx = getRealData(MI::kUxSgx);
  float* uySgy = getRealData(MI::kUySgy);
  float* uzSgz = getRealData(MI::kUzSgz);

  if (rho0ScalarFlag)
  {
    const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
    const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
    const float dividerZ = (simulationDimension == SD::k3D)
                              ? cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz : 1.0f;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dividerX;
      uySgy[i] *= dividerY;
      if (simulationDimension == SD::k3D)
      {
        uzSgz[i] *= dividerZ;
      }
    }
  }
  else
  { // heterogeneous
    const float divider = cudaDeviceConstants.fftDivider * 0.5f;

    const float* dtRho0Sgx = getRealData(MI::kDtRho0Sgx);
    const float* dtRho0Sgy = getRealData(MI::kDtRho0Sgy);
    const float* dtRho0Sgz = getRealData(MI::kDtRho0Sgz);

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dtRho0Sgx[i] * divider;
      uySgy[i] *= dtRho0Sgy[i] * divider;
      if (simulationDimension == SD::k3D)
      {
        uzSgz[i] *= dtRho0Sgz[i] * divider;
      }
    }
  }
}// end of cudaComputeInitialVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface kernel to compute acoustic velocity for initial pressure problem uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeInitialVelocityUniform()
{
  cudaComputeInitialVelocityUniform<simulationDimension, rho0ScalarFlag>
                                   <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeInitialVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = dt ./ rho0_sgx .* dxudxn_sgx .* ifft(ux_sgx);
 *  uy_sgy = dt ./ rho0_sgy .* dyudxn_sgy .* ifft(uy_sgy);
 *  uz_sgz = dt ./ rho0_sgz .* dzudzn_sgz .* ifft(uz_sgz);
 * \endcode
 */
template<SD simulationDimension>
__global__ void cudaComputeInitialVelocityHomogeneousNonuniform()
{
  const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
  const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
  const float dividerZ = (simulationDimension == SD::k3D)
                            ? cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz : 1.0f;

  const float* dxudxnSgx = getRealData(MI::kDxudxnSgx);
  const float* dyudynSgy = getRealData(MI::kDyudynSgy);
  const float* dzudznSgz = getRealData(MI::kDzudznSgz);

  float* uxSgx = getRealData(MI::kUxSgx);
  float* uySgy = getRealData(MI::kUySgy);
  float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords<simulationDimension>(i);

    uxSgx[i] *= dividerX * dxudxnSgx[coords.x];
    uySgy[i] *= dividerY * dyudynSgy[coords.y];

    if (simulationDimension == SD::k3D)
    {
      uzSgz[i] *= dividerZ * dzudznSgz[coords.z];
    }
  }
}// end of cudaComputeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform
 * grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeInitialVelocityHomogeneousNonuniform()
{
  cudaComputeInitialVelocityHomogeneousNonuniform<simulationDimension>
                                                 <<<getSolverGridSize1D(), getSolverBlockSize1D()>>>();

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of computeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

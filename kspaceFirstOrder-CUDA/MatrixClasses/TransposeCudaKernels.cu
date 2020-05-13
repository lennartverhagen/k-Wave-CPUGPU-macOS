/**
 * @file      TransposeCudaKernels.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file for CUDA transpose kernels for 3D FFTs.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      02 August    2019, 14:49 (created) \n
 *            11 February  2020, 16:17 (revised)
 *
 * @copyright Copyright (C) 2019 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <MatrixClasses/TransposeCudaKernels.cuh>
#include <Parameters/CudaDeviceConstants.cuh>

#include <Logger/Logger.h>
#include <Utils/CudaUtils.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get block size for the transposition kernels.
 * @return 3D grid size.
 */
inline dim3 getSolverTransposeBlockSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeBlockSize();
};// end of getSolverTransposeBlockSize()
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get grid size for complex 3D kernels
 * @return 3D grid size
 */
inline dim3 GetSolverTransposeGirdSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeGirdSize();
};// end of getSolverTransposeGirdSize()
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Cuda kernel to transpose a 3D matrix of any dimension sizes in XY planes.\n
 * Every CUDA block in a 1D grid transposes a few slabs.
 * Every CUDA block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the shared memory padding is to eliminate bank
 * conflicts.
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 * @tparam      padding      - Which matrices are padded.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 * @tparam      tilesAtOnce  - How many tiles to transpose at once (4 is default).
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to be of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
template<TransposeCudaKernels::TransposePadding padding,
         int                                    warpSize,
         int                                    tilesAtOnce>
__global__ void cudaTrasnposeReal3DMatrixXY(float*       outputMatrix,
                                            const float* inputMatrix,
                                            const dim3   dimSizes)
{
  // We transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts.
  __shared__ float sharedTile[tilesAtOnce][warpSize][warpSize + 1];

  using TP = TransposeCudaKernels::TransposePadding;
  // Pad input and output dimension by 1 complex element if necessary
  const int nxPadded = ((padding == TP::kInput)  || (padding == TP::kInputOutput))
                         ? 2 * (dimSizes.x / 2 + 1) : dimSizes.x;
  const int nyPadded = ((padding == TP::kOutput) || (padding == TP::kInputOutput))
                         ? 2 * (dimSizes.y / 2 + 1) : dimSizes.y;

  const dim3 tileCount = {dimSizes.x / warpSize + 1, dimSizes.y / warpSize + 1, 1};

  // Run over all slabs. Each CUDA block processes a single slab.
  for (auto slabIdx = blockIdx.x; slabIdx < dimSizes.z; slabIdx += gridDim.x)
  {
    // Calculate offset of the slab
    const float* inputSlab  = inputMatrix  + (nxPadded * dimSizes.y * slabIdx);
          float* outputSlab = outputMatrix + (dimSizes.x * nyPadded * slabIdx);

    dim3 tileIdx = {0, 0, 0};

    // Lambda function to copy data into shared tile.
    auto copyToShared = [=](int y, int x, int row)
    {
      auto globalX = x * warpSize + threadIdx.x;
      auto globalY = y * warpSize + row;

      // If the column is still within the matrix.
      if (globalX < dimSizes.x)
      {
        sharedTile[threadIdx.y][row][threadIdx.x] = inputSlab[globalY * nxPadded + globalX];
      }
    };

    // Lambda function to copy data to output matrix.
    auto copyToOutput = [=](int y, int x, int row)
    {
      auto globalY = y * warpSize + row;
      auto globalX = x * warpSize + threadIdx.x;

      // If the row of the output matrix is still within the matrix.
      if (globalX < dimSizes.y)
      {
        outputSlab[globalY * nyPadded + globalX] = sharedTile[threadIdx.y][threadIdx.x][row];
      }
    };

    // Lambda function to compute whether is the row is within the matrix.
    auto isRowWithinMatrix = [](int tileIdx, int row, int dimension)
    {
      return (row < warpSize) && ((tileIdx * warpSize + row) < dimension);
    };

    // Go over the tiles in the y dimension. Multiple tiles processed simultaneously are under each other.
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // Go over the tiles in the x dimension.
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over the rows in a tile and load data. Process only rows lying within the matrix.
        for (auto row = 0; isRowWithinMatrix(tileIdx.y, row, dimSizes.y); row++)
        {
          copyToShared(tileIdx.y, tileIdx.x, row);
        }// Load data
        __syncwarp();

        // Go over the rows in a transposed tile and store data. Process only rows lying within the matrix.
        for (auto row = 0; isRowWithinMatrix(tileIdx.x, row, dimSizes.x); row++)
        {
          copyToOutput(tileIdx.x, tileIdx.y, row);
        }// Store data
        __syncwarp();

      }// for x dimension
    }// for y dimension
  }// slab
}// end of cudaTrasnposeReal3DMatrixXY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
 * As long as the blockSize.z == 1, the transposition works also for 2D case.
 */
template<TransposeCudaKernels::TransposePadding padding>
void TransposeCudaKernels::trasposeReal3DMatrixXY(float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes)
{
  // Fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  cudaTrasnposeReal3DMatrixXY<padding, 32, 4>
                             <<<GetSolverTransposeGirdSize(), getSolverTransposeBlockSize()>>>
                             (outputMatrix, inputMatrix, dimSizes);

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of trasposeReal3DMatrixXY
//----------------------------------------------------------------------------------------------------------------------

//---------------------------------- Explicit instances of TrasposeReal3DMatrixXY ------------------------------------//
/// Transpose a real 3D matrix in the X-Y direction, input matrix padded, output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input matrix compact, output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input and output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kNone>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInputOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to transpose a 3D matrix in XZ planes of any dimension sizes.\n
 * Every CUDA block in a 1D grid transposes a few slabs.
 * Every CUDA block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the shared memory padding is to eliminate bank
 * conflicts.
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 * @tparam      padding      - Which matrices are padded.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 * @tparam      tilesAtOnce  - How many tiles to transpose at once (4 is default).
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1. \n
 *          Grid has to be organized (N, 1 ,1).
 *
 */
template<TransposeCudaKernels::TransposePadding padding,
         int                                    warpSize,
         int                                    tilesAtOnce>
__global__ void cudaTrasnposeReal3DMatrixXZ(float*       outputMatrix,
                                            const float* inputMatrix,
                                            const dim3   dimSizes)
{
  // We transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts.
  __shared__ float sharedTile[tilesAtOnce][warpSize][ warpSize + 1];

  using TP = TransposeCudaKernels::TransposePadding;
  // Pad input and output dimension by 1 complex element if necessary
  const int nxPadded = ((padding == TP::kInput)  || (padding == TP::kInputOutput))
                         ? 2 * (dimSizes.x / 2 + 1) : dimSizes.x;
  const int nzPadded = ((padding == TP::kOutput) || (padding == TP::kInputOutput))
                         ? 2 * (dimSizes.z / 2 + 1) : dimSizes.z;

  const dim3 tileCount = {dimSizes.x / warpSize + 1, dimSizes.z / warpSize + 1, 1};

  // Run over all XZ slabs. Each CUDA block processes a single slab.
  for (auto row = blockIdx.x; row < dimSizes.y; row += gridDim.x )
  {
    dim3 tileIdx = {0, 0, 0};

    // Lambda function to copy data into shared tile.
    auto copyToShared = [=](int y, int x, int slab)
    {
      auto globalX = x * warpSize + threadIdx.x;
      auto globalY = y * warpSize + slab;

      if (globalX < dimSizes.x)
      {
        sharedTile[threadIdx.y][slab][threadIdx.x]
              = inputMatrix[globalY * nxPadded * dimSizes.y + row * nxPadded + globalX];
      }
    };

    // Lambda function to copy data to output matrix.
    auto copyToOutput = [=](int y, int x, int slab)
    {
      auto globalY = y * warpSize + slab;
      auto globalX = x * warpSize + threadIdx.x;

      if (globalX < dimSizes.z)
      {
        outputMatrix[globalY * dimSizes.y * nzPadded + (row * nzPadded) + globalX]
              = sharedTile[threadIdx.y][threadIdx.x][slab];
      }
    };

    // Lambda function to compute whether is the row is within the matrix.
    auto isSlabWithinMatrix = [](int tileIdx, int slab, int dimension)
    {
      return (slab < warpSize) && ((tileIdx * warpSize + slab) < dimension);
    };

    // Go over all all tiles in the XZ slab. Transpose multiple slabs at the same time (one per Z)
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // Go over the tiles in the x dimension.
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over the slabs in a tile and load data. Process only slabs lying within the matrix.
        for (auto slab = 0; isSlabWithinMatrix(tileIdx.y, slab, dimSizes.z); slab++)
        {
          copyToShared(tileIdx.y, tileIdx.x, slab);
        }// Load data
        __syncwarp();

        // Go over the slabs in a transposed tile and store data. Process only slabs lying within the matrix.
        for (auto slab = 0; isSlabWithinMatrix(tileIdx.x, slab, dimSizes.x); slab++)
        {
          copyToOutput(tileIdx.x, tileIdx.y, slab);
        }// Store data
        __syncwarp();
      }// for x dimension
    }// for y dimension
  }// slab
}// end of cudaTrasnposeReal3DMatrixXZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Transpose a real 3D matrix in the X-Z direction. It is done out-of-place.
 */
template<TransposeCudaKernels::TransposePadding padding>
void TransposeCudaKernels::trasposeReal3DMatrixXZ(float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes)
{
  // Fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  cudaTrasnposeReal3DMatrixXZ<padding, 32, 4>
                              <<<GetSolverTransposeGirdSize(), getSolverTransposeBlockSize()>>>
                             (outputMatrix, inputMatrix, dimSizes);

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of trasposeReal3DMatrixXZ
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------- Explicit instances of TrasposeReal3DMatrixXZ -------------------------------------//
/// Transpose a real 3D matrix in the X-Z direction, input matrix padded, output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input matrix compact, output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kNone>
                                                (float*       outputMatrix,
                                                 const float* inputMatrix,
                                                 const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInputOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------

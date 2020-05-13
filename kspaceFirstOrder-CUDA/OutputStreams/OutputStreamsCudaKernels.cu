/**
 * @file      OutputStreamsCudaKernels.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of cuda kernels used for data sampling (output streams).
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      27 January   2015, 17:21 (created) \n
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

#include <cuda.h>
#include <cuda_runtime.h>

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>
#include <Utils/CudaUtils.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Global routines ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Get Sampler CUDA Block size.
 * @return CUDA block size.
 */
inline int getSamplerBlockSize()
{
  return Parameters::getInstance().getCudaParameters().getSamplerBlockSize1D();
}// end of getSamplerBlockSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get sampler CUDA grid size.
 * @return CUDA grid size.
 */
inline int getSamplerGridSize()
{
  return Parameters::getInstance().getCudaParameters().getSamplerGridSize1D();
}// end of getSamplerGridSize
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Index mask sampling --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to sample data based on index sensor mask. The operator is given by the template parameter.
 *
 * @param [out] samplingBuffer - Buffer to sample data in.
 * @param [in]  sourceData     - Source matrix.
 * @param [in]  sensorData     - Sensor mask.
 * @param [in]  nSamples       - Number of sampled points.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
__global__ void cudaSampleIndex(float*        samplingBuffer,
                                const float*  sourceData,
                                const size_t* sensorData,
                                const size_t  nSamples)
{
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kNone:
      {
        samplingBuffer[i] = sourceData[sensorData[i]];
        break;
      }

      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[sensorData[i]] * sourceData[sensorData[i]]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }
    }// switch
  }// for
}// end of cudaSampleIndex
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample the source matrix using the index sensor mask and store data in buffer.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleIndex(float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples)
{
  cudaSampleIndex<reduceOp>
                 <<<getSamplerGridSize(),getSamplerBlockSize()>>>
                 (samplingBuffer, sourceData, sensorData, nSamples);

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sampleIndex
//----------------------------------------------------------------------------------------------------------------------

/// Sample the source matrix using the index sensor mask, no post-processing.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kNone>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the root mean square.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kRms>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the maximum.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kMax>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the minimum.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kMin>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Cuboid mask sampling -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Transform 3D coordinates within the cuboid into 1D coordinates within the matrix being sampled.
 *
 * @param [in] cuboidIdx         - Cuboid index.
 * @param [in] topLeftCorner     - Top left corner.
 * @param [in] bottomRightCorner - Bottom right corner.
 * @param [in] matrixSize        - Size of the matrix being sampled.
 * @return 1D index into the matrix being sampled.
 */
inline __device__ size_t transformCoordinates(const size_t cuboidIdx,
                                              const dim3&  topLeftCorner,
                                              const dim3&  bottomRightCorner,
                                              const dim3&  matrixSize)
{
  dim3 localPosition;
  // Calculate the cuboid size
  dim3 cuboidSize(bottomRightCorner.x - topLeftCorner.x + 1,
                  bottomRightCorner.y - topLeftCorner.y + 1,
                  bottomRightCorner.z - topLeftCorner.z + 1);

  // Find coordinates within the cuboid
  size_t slabSize = cuboidSize.x * cuboidSize.y;
  localPosition.z =  cuboidIdx / slabSize;
  localPosition.y = (cuboidIdx % slabSize) / cuboidSize.x;
  localPosition.x = (cuboidIdx % slabSize) % cuboidSize.x;

  // Transform the coordinates to the global dimensions
  dim3 globalPosition(localPosition);
  globalPosition.z += topLeftCorner.z;
  globalPosition.y += topLeftCorner.y;
  globalPosition.x += topLeftCorner.x;

  // Calculate 1D index
  return (globalPosition.z * matrixSize.x * matrixSize.y +
          globalPosition.y * matrixSize.x +
          globalPosition.x);
}// end of transformCoordinates
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to sample data inside one cuboid, operation is selected by a template parameter.

 * @param [out] samplingBuffer    - Buffer to sample data in.
 * @param [in]  sourceData        - Source matrix.
 * @param [in]  topLeftCorner     - Top left corner of the cuboid.
 * @param [in]  bottomRightCorner - Bottom right corner of the cuboid.
 * @param [in]  matrixSize        - Dimension sizes of the matrix being sampled.
 * @param [in]  nSamples          - Number of grid points inside the cuboid.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
__global__ void cudaSampleCuboid(float*       samplingBuffer,
                                 const float* sourceData,
                                 const dim3   topLeftCorner,
                                 const dim3   bottomRightCorner,
                                 const dim3   matrixSize,
                                 const size_t nSamples)
{
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    auto Position = transformCoordinates(i, topLeftCorner, bottomRightCorner, matrixSize);
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kNone:
      {
        samplingBuffer[i] = sourceData[Position];
        break;
      }

      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[Position] * sourceData[Position]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[Position]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[Position]);
        break;
      }
    }// switch
  }// for
}// end of cudaSampleCuboid
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample data inside one cuboid and store it to buffer. The operation is given in the template parameter.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleCuboid(float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples)
{
  cudaSampleCuboid<reduceOp>
                  <<<getSamplerGridSize(),getSamplerBlockSize()>>>
                  (samplingBuffer, sourceData, topLeftCorner, bottomRightCorner, matrixSize, nSamples);
  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sampleCuboid
//----------------------------------------------------------------------------------------------------------------------

///Sample data inside one cuboid and store it to buffer, no post-processing.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kNone>
                                           (float*       samplingBuffer,
                                            const float*  sourceData,
                                            const dim3    topLeftCorner,
                                            const dim3    bottomRightCorner,
                                            const dim3    matrixSize,
                                            const size_t  nSamples);
/// Sample data inside one cuboid and store it to buffer, take the root mean square.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kRms>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
/// Sample data inside one cuboid and store it to buffer, take the maximum.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kMax>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
/// Sample data inside one cuboid and store it to buffer, take the minimum.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kMin>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Whole domain based sampling --------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to sample and aggregate the source matrix on the whole domain and apply a reduce operator.
 *
 * @param [in,out] samplingBuffer - Buffer to sample data in.
 * @param [in]     sourceData     - Source matrix.
 * @param [in]     nSamples       - Number of sampled points.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
__global__ void cudaSampleAll(float*       samplingBuffer,
                              const float* sourceData,
                              const size_t nSamples)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[i] * sourceData[i]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[i]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[i]);
        break;
      }
    }
  }
}// end of cudaSampleAll
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample and the whole domain and apply a defined operator.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleAll(float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples)
{
  cudaSampleAll<reduceOp>
               <<<getSamplerGridSize(),getSamplerBlockSize()>>>
               (samplingBuffer, sourceData, nSamples);

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of sampleMaxAll
//----------------------------------------------------------------------------------------------------------------------

/// Sample and the whole domain and apply a defined operator, take the root mean square.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kRms>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
/// Sample and the whole domain and apply a defined operator, take the maximum.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kMax>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
/// Sample and the whole domain and apply a defined operator, take the minimum.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kMin>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Post-processing -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to apply post-processing for RMS.
 *
 * @param [in, out] samplingBuffer - Buffer to apply post-processing on.
 * @param [in]      scalingCoeff   - Scaling coeficinet for RMS.
 * @param [in]      nSamples       - Number of elements.
 */
__global__ void cudaPostProcessingRms(float*       samplingBuffer,
                                      const float  scalingCoeff,
                                      const size_t nSamples)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    samplingBuffer[i] = sqrt(samplingBuffer[i] * scalingCoeff);
  }
}// end of cudaPostProcessingRMS
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate post-processing for RMS.
 */
void OutputStreamsCudaKernels::postProcessingRms(float*       samplingBuffer,
                                                 const float  scalingCoeff,
                                                 const size_t nSamples)
{
  cudaPostProcessingRms<<<getSamplerGridSize(),getSamplerBlockSize()>>>
                       (samplingBuffer, scalingCoeff, nSamples);

  // Check for errors
  cudaCheckErrors(cudaGetLastError());
}// end of postProcessingRms
//----------------------------------------------------------------------------------------------------------------------

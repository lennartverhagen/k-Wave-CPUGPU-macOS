/**
 * @file      CudaMatrixContainer.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file defining the cuda matrix container used in cuda kernels.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      26 August    2019, 11:07 (created) \n
 *            11 February  2020, 16:10 (revised)
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

#ifndef CUDA_MATRIX_CONTAINER_CUH
#define CUDA_MATRIX_CONTAINER_CUH

#include<cuComplex.h>

#include <Containers/MatrixRecord.h>
#include <Containers/MatrixContainer.h>

/**
 * @class   CudaMatrixContainer
 * @brief   This container holds pointers to device matrices used in CUDA kernels.
 *
 * @details Since STL containers are not allowed in CUDA, the raw pointers are copied into an array of a predefined
 *          size. This array can hold all possible matrices. If some of them is not present in the simulation,
 *          the pointer is set to nullptr.
 *
 *          The container is initialized on the host side after the MatrixContainer has been allocated and the
 *          pointers to device data transferred afterwards.
 *
 * @tparam  size - Size of the container.
 */
template<size_t size>
class CudaMatrixContainer
{
 public:
    /// Default constructor.
    __host__ CudaMatrixContainer();

    /**
     * @brief Copy matrix records (raw data pointers) inside this container on the host size.
     * @param [in] matrixIdx    - Identifier of the matrix.
     * @param [in] matrixRecord - Matrix record holding matrix type, data, etc.
     */
    __host__ void copyMatrixRecord(const MatrixContainer::MatrixIdx matrixIdx,
                                   const MatrixRecord&              matrixRecord);

    /// Upload container content into device constant memory.
    __host__ void copyToDevice();

    /**
     * @brief  Return pointer to float matrix data.
     * @param  [in] matrixIdx - Matrix index.
     * @return Pointer to device data.
     */
    __device__ float* getRealData(MatrixContainer::MatrixIdx matrixIdx)
    {
      return mMatrixContainer[static_cast<int>(matrixIdx)].floatData;
    };
    /**
     * @brief  Return pointer to complex matrix data.
     * @param  [in] matrixIdx - Matrix index.
     * @return Pointer to device data.
     */
    __device__ cuFloatComplex* getComplexData(MatrixContainer::MatrixIdx matrixIdx)
    {
      return mMatrixContainer[static_cast<int>(matrixIdx)].complexData;
    };
    /**
     * @brief  Return pointer to index matrix data.
     * @param  [in] matrixIdx - Matrix index.
     * @return Pointer to device data.
     */
    __device__ size_t* getIndexData(MatrixContainer::MatrixIdx matrixIdx)
    {
      return mMatrixContainer[static_cast<int>(matrixIdx)].indexData;
    };

 private:
    /**
     * Union to store pointers to matrix data.
     */
    union DeviceMatrixRecord
    {
      /// Pointer to device RealMatrix data.
      float*          floatData;
      /// Pointer to device ComplexMatrix and CufftComplexMatrix data.
      cuFloatComplex* complexData;
      /// Pointer to device IndexMatrix data.
      size_t*         indexData;
    };

    /// Container holding all device pointers accessed by a MatrixIdx converted to unsigned int64.
    DeviceMatrixRecord mMatrixContainer[size];
};// end of CudaMatrixContainer
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUDA_MATRIX_CONTAINER_CUH */

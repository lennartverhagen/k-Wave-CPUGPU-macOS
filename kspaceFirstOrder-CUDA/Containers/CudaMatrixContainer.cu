/**
 * @file      CudaMatrixContainer.cu
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the cuda matrix container used in cuda kernels.
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

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Containers/CudaMatrixContainer.cuh>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// Forward declaration for the compiler to generate necessary template.
template class CudaMatrixContainer<MatrixContainer::getMatrixIdxCount()>;

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Variables -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @var      cudaMatrixContainer
 * @brief    This variable holds pointer data to device matrices present in MatrixContainer.
 * @details  This variable is imported as extern into other CUDA units.
 */
__constant__ CudaMatrixContainer<MatrixContainer::getMatrixIdxCount()> cudaMatrixContainer;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor clearing the container on the host side.
 */
template<size_t size>
__host__ CudaMatrixContainer<size>::CudaMatrixContainer()
{
  // Data is cleared only on the host size
  #ifndef  __CUDA_ARCH__
    for (size_t i = 0; i < size; i++)
    {
      mMatrixContainer[i].floatData = nullptr;
    }
  #endif
}// end of default constructor.
//----------------------------------------------------------------------------------------------------------------------

 /**
  * Copy matrix records (raw data pointers) inside this container on the host size.
  */
template<size_t size>
__host__ void CudaMatrixContainer<size>::copyMatrixRecord(const MatrixContainer::MatrixIdx matrixIdx,
                                                          const MatrixRecord&              matrixRecord)
{
  using MT = MatrixRecord::MatrixType;

  // Copy device pointers to the array.
  switch (matrixRecord.matrixType)
  {
    case MT::kReal:
    {
      mMatrixContainer[int(matrixIdx)].floatData
        = static_cast<RealMatrix*>(matrixRecord.matrixPtr)->getDeviceData();
      break;
    }

    case MT::kComplex:
    {
      mMatrixContainer[int(matrixIdx)].complexData
        = static_cast<ComplexMatrix*>(matrixRecord.matrixPtr)->getComplexDeviceData();
      break;
    }

    case MT::kCufft:
    {
      mMatrixContainer[int(matrixIdx)].complexData
        = static_cast<CufftComplexMatrix*>(matrixRecord.matrixPtr)->getComplexDeviceData();
      break;
    }

    case MT::kIndex:
    {
      mMatrixContainer[int(matrixIdx)].indexData
        = static_cast<IndexMatrix*>(matrixRecord.matrixPtr)->getDeviceData();
      break;
    }

    default:
    {
      throw std::invalid_argument(Logger::formatMessage(kErrFmtBadMatrixType, matrixRecord.matrixName.c_str()));
    }
  }
}// end of copyMatrixRecords
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy the structure with simulation constants to the CUDA constant memory.
 */
template<size_t size>
__host__ void CudaMatrixContainer<size>::copyToDevice()
{
  cudaCheckErrors(cudaMemcpyToSymbol(cudaMatrixContainer, this, sizeof(cudaMatrixContainer)));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

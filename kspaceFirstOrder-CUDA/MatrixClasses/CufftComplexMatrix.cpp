/**
 * @file      CufftComplexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the class implementing various and 1D FFTs using the cuFFT interface.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      09 August    2011, 13:10 (created) \n
 *            11 February  2020, 16:17 (revised)
 *
 * @copyright Copyright (C) 2011 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <string>
#include <stdexcept>
#include <cufft.h>

#include <MatrixClasses/CufftComplexMatrix.h>
#include <MatrixClasses/TransposeCudaKernels.cuh>
#include <MatrixClasses/RealMatrix.h>
#include <Logger/Logger.h>
#include <KSpaceSolver/SolverCudaKernels.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

cufftHandle CufftComplexMatrix::sR2CFftPlanND = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlanND = cufftHandle();

cufftHandle CufftComplexMatrix::sR2CFftPlan1DX = cufftHandle();
cufftHandle CufftComplexMatrix::sR2CFftPlan1DY = cufftHandle();
cufftHandle CufftComplexMatrix::sR2CFftPlan1DZ = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DX = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DY = cufftHandle();
cufftHandle CufftComplexMatrix::sC2RFftPlan1DZ = cufftHandle();


/**
 * Error message for the CufftComplexMatrix FFT class.
 */
std::map<cufftResult, ErrorMessage> CufftComplexMatrix::sCufftErrorMessages
{
  {CUFFT_INVALID_PLAN             , kErrFmtCufftInvalidPlan},
  {CUFFT_ALLOC_FAILED             , kErrFmtCufftAllocFailed},
  {CUFFT_INVALID_TYPE             , kErrFmtCufftInvalidType},
  {CUFFT_INVALID_VALUE            , kErrFmtCufftInvalidValue},
  {CUFFT_INTERNAL_ERROR           , kErrFmtCuFFTInternalError},
  {CUFFT_EXEC_FAILED              , kErrFmtCufftExecFailed},
  {CUFFT_SETUP_FAILED             , kErrFmtCufftSetupFailed},
  {CUFFT_INVALID_SIZE             , kErrFmtCufftInvalidSize},
  {CUFFT_UNALIGNED_DATA           , kErrFmtCufftUnalignedData},
  {CUFFT_INCOMPLETE_PARAMETER_LIST, kErrFmtCufftIncompleteParaterList},
  {CUFFT_INVALID_DEVICE           , kErrFmtCufftInvalidDevice},
  {CUFFT_PARSE_ERROR              , kErrFmtCufftParseError},
  {CUFFT_NO_WORKSPACE             , kErrFmtCufftNoWorkspace},
  {CUFFT_NOT_IMPLEMENTED          , kErrFmtCufftNotImplemented},
  {CUFFT_LICENSE_ERROR            , kErrFmtCufftLicenseError},
  {CUFFT_NOT_SUPPORTED            , kErrFmtCufftNotSupported}
};
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create an cuFFT plan for 2D/3D Real-to-Complex transform.
 */
void CufftComplexMatrix::createR2CFftPlanND(const DimensionSizes& inMatrixDims)
{
  cufftResult cufftError;
  if (Parameters::getInstance().isSimulation3D())
  {
    cufftError= cufftPlan3d(&sR2CFftPlanND,
                            static_cast<int>(inMatrixDims.nz),
                            static_cast<int>(inMatrixDims.ny),
                            static_cast<int>(inMatrixDims.nx),
                            CUFFT_R2C);
  }
  else
  {
    cufftError= cufftPlan2d(&sR2CFftPlanND,
                            static_cast<int>(inMatrixDims.ny),
                            static_cast<int>(inMatrixDims.nx),
                            CUFFT_R2C);
  }

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateR2CFftPlanND);
  }
}// end of createR2CFftPlanND
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 2D/3D Complex-to-Real transform.
 */
void CufftComplexMatrix::createC2RFftPlanND(const DimensionSizes& outMatrixDims)
{
  cufftResult cufftError;
  if (Parameters::getInstance().isSimulation3D())
  {
    cufftError = cufftPlan3d(&sC2RFftPlanND,
                             static_cast<int>(outMatrixDims.nz),
                             static_cast<int>(outMatrixDims.ny),
                             static_cast<int>(outMatrixDims.nx),
                             CUFFT_C2R);
  }
  else
  {
    cufftError = cufftPlan2d(&sC2RFftPlanND,
                             static_cast<int>(outMatrixDims.ny),
                             static_cast<int>(outMatrixDims.nx),
                             CUFFT_C2R);
  }

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateC2RFftPlanND);
  }
}// end of createC2RFftPlan3D
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DX Real-to-Complex transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations.
 */
void CufftComplexMatrix::createR2CFftPlan1DX(const DimensionSizes& inMatrixDims)
{
  // Set dimensions
  const int nx   = static_cast<int>(inMatrixDims.nx);
  const int ny   = static_cast<int>(inMatrixDims.ny);
  const int nz   = static_cast<int>(inMatrixDims.nz);
  const int nxR = ((nx / 2) + 1);

  // Set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since running out-of-place, no padding is needed.
  int inembed[] = {nx};
  int istride   = 1;
  int idist     = nx;

  int onembed[] = {nxR};
  int ostride   = 1;
  int odist     = nxR;

  int batch = ny * nz;

  // Plan the FFT
  cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DX, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DX);
  }
}// end of createR2CFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DY Real-to-Complex transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations.
 */
void CufftComplexMatrix::createR2CFftPlan1DY(const DimensionSizes& inMatrixDims)
{
  // Set dimensions
  const int nx   = static_cast<int> (inMatrixDims.nx);
  const int ny   = static_cast<int> (inMatrixDims.ny);
  const int nz   = static_cast<int> (inMatrixDims.nz);
  const int nyR = ((ny / 2) + 1);

  // Set up rank and strides
  int rank = 1;
  int n[] = {ny};

  // The input matrix is transposed with every row padded by a single element.
  int inembed[] = {2 * nyR};
  int istride   = 1;
  int idist     = 2 * nyR;

  int onembed[] = {nyR};
  int ostride   = 1;
  int odist     = nyR;

  int batch =  nx * nz;

  cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DY, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_R2C, batch);

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DY);
  }
}// end of createR2CFftPlan1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DZ Real-to-Complex transform. This routine throws en exception when called for 2D simulation.
 */
void CufftComplexMatrix::createR2CFftPlan1DZ(const DimensionSizes& inMatrixDims)
{
  if (Parameters::getInstance().isSimulation2D())
  {
    // Throw error when this routine is called for 2D simulations
    throw std::runtime_error(kErrFmtCannotCallR2CFftPlan1DZfor2D);
  }
  else
  {
    const int nx   = static_cast<int> (inMatrixDims.nx);
    const int ny   = static_cast<int> (inMatrixDims.ny);
    const int nz   = static_cast<int> (inMatrixDims.nz);
    const int nzR = ((nz / 2) + 1);

    // Set up rank and strides
    int rank = 1;
    int n[] = {nz};

    // The input matrix is transposed with every row padded by a single element.
    int inembed[] = {2 * nzR};
    int istride   = 1;
    int idist     = 2 * nzR;

    int onembed[] = {nzR};
    int ostride   = 1;
    int odist     = nzR;

    int batch =  nx * ny;

    cufftResult_t cufftError = cufftPlanMany(&sR2CFftPlan1DZ, rank, n,
                                             inembed, istride, idist,
                                             onembed, ostride, odist,
                                             CUFFT_R2C, batch);

    if (cufftError != CUFFT_SUCCESS)
    {
      throwCufftException(cufftError, kErrFmtCreateR2CFftPlan1DZ);
    }
  }
}// end of createR2CFftPlan1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DX Complex-to-Real transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations.
 */
void CufftComplexMatrix::createC2RFftPlan1DX(const DimensionSizes& outMatrixDims)
{
  // Set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nxR = ((nx / 2) + 1);

  // Set up rank and strides
  int rank = 1;
  int n[] = {nx};

  // Since runs out-of-place no padding is needed.
  int inembed[] = {nxR};
  int istride   = 1;
  int idist     = nxR;

  int onembed[] = {nx};
  int ostride   = 1;
  int odist     = nx;

  int batch = ny * nz;

  cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DX, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DX);
  }
}// end of createC2RFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DY Complex-to-Real transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations.
 */
void CufftComplexMatrix::createC2RFftPlan1DY(const DimensionSizes& outMatrixDims)
{
  // Set dimensions
  const int nx   = static_cast<int> (outMatrixDims.nx);
  const int ny   = static_cast<int> (outMatrixDims.ny);
  const int nz   = static_cast<int> (outMatrixDims.nz);
  const int nyR = ((ny / 2) + 1);

  // Set up rank and strides
  int rank = 1;
  int n[] = {ny};

  int inembed[] = {nyR};
  int istride   = 1;
  int idist     = nyR;

  // The output matrix is transposed with every row padded by a single element.
  int onembed[] = {2 * nyR};
  int ostride   = 1;
  int odist     = 2 * nyR;

  int batch =  nx * nz;

  cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DY, rank, n,
                                           inembed, istride, idist,
                                           onembed, ostride, odist,
                                           CUFFT_C2R, batch);

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DY);
  }
}// end of createC2RFftPlan1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create cuFFT plan for 1DZ Complex-to-Real transform. This routine throws en exception when called for 2D simulation.
 */
void CufftComplexMatrix::createC2RFftPlan1DZ(const DimensionSizes& outMatrixDims)
{
  if (Parameters::getInstance().isSimulation2D())
  {
    // Throw error when this routine is called for 2D simulations
    throw std::runtime_error(kErrFmtCannotCallR2CFftPlan1DZfor2D);
  }
  else
  {
    // Set dimensions
    const int nx   = static_cast<int> (outMatrixDims.nx);
    const int ny   = static_cast<int> (outMatrixDims.ny);
    const int nz   = static_cast<int> (outMatrixDims.nz);
    const int nzR = ((nz / 2) + 1);

    // Set up rank and strides
    int rank = 1;
    int n[] = {nz};

    int inembed[] = {nzR};
    int istride   = 1;
    int idist     = nzR;

    // The output matrix is transposed with every row padded by a single element.
    int onembed[] = {2 * nzR};
    int ostride   = 1;
    int odist     = 2 * nzR;

    int batch =  nx * ny;

    cufftResult_t cufftError = cufftPlanMany(&sC2RFftPlan1DZ, rank, n,
                                             inembed, istride, idist,
                                             onembed, ostride, odist,
                                             CUFFT_C2R, batch);

    if (cufftError != CUFFT_SUCCESS)
    {
      throwCufftException(cufftError, kErrFmtCreateC2RFftPlan1DZ);
    }
  }
}// end of createC2RFftPlan1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destroy all static plans created by the application.
 */
void CufftComplexMatrix::destroyAllPlansAndStaticData()
{
  cufftResult_t cufftError;

  if (sR2CFftPlanND)
  {
    cufftError = cufftDestroy(sR2CFftPlanND);
    sR2CFftPlanND = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlanND);
  }

  if (sC2RFftPlanND)
  {
    cufftError = cufftDestroy(sC2RFftPlanND);
    sC2RFftPlanND = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlanND);
  }

  if (sR2CFftPlan1DX)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DX);
    sR2CFftPlan1DX = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DX);
  }

  if (sR2CFftPlan1DY)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DY);
    sR2CFftPlan1DY = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DY);
  }

  if (sR2CFftPlan1DZ)
  {
    cufftError = cufftDestroy(sR2CFftPlan1DZ);
    sR2CFftPlan1DZ = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyR2CFftPlan1DZ);
  }

  if (sC2RFftPlan1DX)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DX);
    sC2RFftPlan1DX = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DX);
  }

  if (sC2RFftPlan1DY)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DY);
    sC2RFftPlan1DY = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DY);
  }

  if (sC2RFftPlan1DZ)
  {
    cufftError = cufftDestroy(sC2RFftPlan1DZ);
    sC2RFftPlan1DZ = cufftHandle();
    if (cufftError != CUFFT_SUCCESS) throwCufftException(cufftError, kErrFmtDestroyC2RFftPlan1DZ);
  }

  // clear static data
  sCufftErrorMessages.clear();
}// end of destroyAllPlansAndStaticData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place (2D/3D) Real-to-Complex transform.
 */
void CufftComplexMatrix::computeR2CFftND(RealMatrix& inMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlanND,
                                          static_cast<cufftReal*>(inMatrix.getDeviceData()),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteR2CFftPlanND);
  }
}// end of computeR2CFft3D
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place (2D/3D) Complex-to-Real transform.
 */
void CufftComplexMatrix::computeC2RFftND(RealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlanND,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(outMatrix.getDeviceData()));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteC2RFftPlanND);
  }
}// end of computeC2RFft3D
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DX Real-to-Complex transform.
 */
void CufftComplexMatrix::computeR2CFft1DX(RealMatrix& inMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DX,
                                          static_cast<cufftReal*>(inMatrix.getDeviceData()),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DX);
  }
}// end of computeR2CFft1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex transform. The matrix is first X<->Y transposed
 * followed by the 1D FFT. The matrix is left in the transposed format. \n
 *
 * Since nz == 1 in the 2D case, there's no need to modify this routine for 2D simulations.
 */
void CufftComplexMatrix::computeR2CFft1DY(RealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

  TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kOutput>
                                              (mDeviceData,
                                               inMatrix.getDeviceData(),
                                               dimSizes);

  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // The FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DY,
                                          static_cast<cufftReal*>(mDeviceData),
                                          reinterpret_cast<cufftComplex*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DY);
  }
}// end of computeR2CFft1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DZ Real-to-Complex transform. This routine throws en exception when called for 2D
 * simulation.
 */
void CufftComplexMatrix::computeR2CFft1DZ(RealMatrix& inMatrix)
{
  if (Parameters::getInstance().isSimulation3D())
  {
    /// Transpose a real 3D matrix in the X-Z direction
    dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                  static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                  static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

    TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kOutput>
                                                (mDeviceData,
                                                 inMatrix.getDeviceData(),
                                                 dimSizes);

    // Compute forward cuFFT (if the plan does not exist, it also returns error).
    // The FFT is calculated in-place (may be a bit slower than out-of-place, however
    // it does not request additional transfers and memory).
    cufftResult_t cufftError = cufftExecR2C(sR2CFftPlan1DZ,
                                            static_cast<cufftReal*>(mDeviceData),
                                            reinterpret_cast<cufftComplex*>(mDeviceData));

    if (cufftError != CUFFT_SUCCESS)
    {
      throwCufftException(cufftError, kErrFmtExecuteR2CFftPlan1DZ);
    }
  }
  else
  {
    throwCufftException(CUFFT_INVALID_PLAN, kErrFmtExecuteR2CFftPlan1DZ);
  }
}// end of computeR2CFft1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DX Real-to-Complex transform.
 */
void CufftComplexMatrix::computeC2RFft1DX(RealMatrix& outMatrix)
{
  // Compute inverse cuFFT (if the plan does not exist, it also returns error)
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DX,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(outMatrix.getDeviceData()));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DX);
  }
}// end of computeC2RFft1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DY Real-to-Complex transform.
 * The matrix is taken in the transposed format and transposed at the end into a natural form. \n
 *
 * Since nz == 1 in the 2D case, there's no need to modify this routine for 2D simulations.
 */
void CufftComplexMatrix::computeC2RFft1DY(RealMatrix& outMatrix)
{
  // Compute forward cuFFT (if the plan does not exist, it also returns error).
  // The FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DY,
                                          reinterpret_cast<cufftComplex*>(mDeviceData),
                                          static_cast<cufftReal*>(mDeviceData));

  if (cufftError != CUFFT_SUCCESS)
  {
    throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DY);
  }

  /// Transpose a real 3D matrix back in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nz));

  TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInput>
                                              (outMatrix.getDeviceData(),
                                               mDeviceData,
                                               dimSizes);
}// end of computeC2RFft1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex transform. This routine throws en exception when called for 2D
 * simulation.
 */
void CufftComplexMatrix::computeC2RFft1DZ(RealMatrix& outMatrix)
{
  if (Parameters::getInstance().isSimulation3D())
  {
    // Compute forward cuFFT (if the plan does not exist, it also returns error).
    // The FFT is calculated in-place (may be a bit slower than out-of-place, however
    // it does not request additional transfers and memory).
    cufftResult_t cufftError = cufftExecC2R(sC2RFftPlan1DZ,
                                            reinterpret_cast<cufftComplex*>(mDeviceData),
                                            static_cast<cufftReal*>(mDeviceData));

    if (cufftError != CUFFT_SUCCESS)
    {
      throwCufftException(cufftError, kErrFmtExecuteC2RFftPlan1DZ);
    }

    /// Transpose a real 3D matrix in the Z<->X direction
    dim3 DimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().nz),
                  static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                  static_cast<unsigned int>(outMatrix.getDimensionSizes().nx));

    TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInput>
                                                (outMatrix.getDeviceData(),
                                                 getDeviceData(),
                                                 DimSizes);
  }
  else
  {
    throwCufftException(CUFFT_INVALID_PLAN, kErrFmtExecuteC2RFftPlan1DZ);
  }
}// end of computeC2RFft1DZ
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Throw cuda FFT exception.
 */
void CufftComplexMatrix::throwCufftException(const cufftResult  cufftError,
                                             const std::string& transformTypeName)
{
  std::string errMsg;
  if (sCufftErrorMessages.find(cufftError) != sCufftErrorMessages.end())
  {
    errMsg = Logger::formatMessage(sCufftErrorMessages[cufftError], transformTypeName.c_str());
  }
  else // Unknown error
  {
    errMsg = Logger::formatMessage(kErrFmtCufftUnknownError, transformTypeName.c_str());
  }

  // Throw exception
  throw std::runtime_error(errMsg);
}// end of throwCufftException
//----------------------------------------------------------------------------------------------------------------------
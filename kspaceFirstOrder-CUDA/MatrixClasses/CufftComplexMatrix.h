/**
 * @file      CufftComplexMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the class implementing various FFTs using the cuFFT interface.
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

#ifndef CUFFT_COMPLEX_MATRIX_H
#define CUFFT_COMPLEX_MATRIX_H

#include <map>
#include <cufft.h>

#include <MatrixClasses/ComplexMatrix.h>
#include <Logger/ErrorMessages.h>

/**
 * @class   CufftComplexMatrix
 * @brief   Class implementing ND and 1D Real-To-Complex and Complex-To-Real transforms using cuFFT interface.
 * @details Class implementing a single ND (3D, 2D) and many 1D Real-To-Complex and Complex-To-Real transforms
 *          using FFTW interface.
 * \li If the matrix is 3D, the ND transform is 3D and the batch of 1D goes over the second and third dimension.
 * \li If the matrix is 2D, the ND transform is 2D and the batch of 1D goes over the second dimension.

 */
class CufftComplexMatrix : public ComplexMatrix
{
  public:
    /// Default constructor not allowed.
    CufftComplexMatrix() = delete;
    /**
     * @brief Constructor, inherited from ComplexMatrix.
     * @param [in] dimensionSizes - Dimension sizes of the matrix.
     */
    CufftComplexMatrix(const DimensionSizes& dimensionSizes) : ComplexMatrix(dimensionSizes) { };
    /// Copy constructor not allowed.
    CufftComplexMatrix(const CufftComplexMatrix&) = delete;
    /// Destructor (Inherited from ComplexMatrix).
    virtual ~CufftComplexMatrix() = default;

    /// Operator = is not allowed.
    CufftComplexMatrix& operator=(const CufftComplexMatrix&) = delete;

    /**
     * @brief Create cuFFT plan for 2D/3D Real-to-Complex transform.
     * @param [in] inMatrixDims  - The dimension sizes of the input matrix.
     * @throw std::runtime_error - If the plan can't be created.
     */
    static void createR2CFftPlanND(const DimensionSizes& inMatrixDims);
    /**
     * @brief Create cuFFT plan for 2D/3D Complex-to-Real transform.
     * @param [in] outMatrixDims  - the dimension sizes of the output matrix.
     * @throw std::runtime_error  - If the plan can't be created.
     */
    static void createC2RFftPlanND(const DimensionSizes& outMatrixDims);

    /**
     * @brief   Create cuFFT plan for 1DX Real-to-Complex transform.
     * @details This version doesn't need any scratch place for planning. All 1D transforms are done in a
     *          single batch (no transpose needed) and in out-of-place manner.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] inMatrixDims  - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createR2CFftPlan1DX(const DimensionSizes& inMatrixDims);
    /**
     * @brief   Create cuFFT plan for 1DY Real-to-Complex transform.
     * @details This version doesn't need any scratch place for planning. All 1D transforms are done in a single
     *          batch. Data is transposed and padded according to the cuFFT data layout before the
     *          transform. The FFT is done in-place.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] inMatrixDims  - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createR2CFftPlan1DY(const DimensionSizes& inMatrixDims);
    /**
     * @brief   Create cuFFT plan for 1DZ Real-to-Complex transform.
     * @details This version doesn't need any scratch place for planning.  All 1D transforms are done in a single
     *          batch. Data has to be transposed and padded according to the cuFFT data layout before the
     *          transform. The FFT is done in-place.
     *
     *          This routine can only be used for both 3D simulations.
     *
     * @param   [in] inMatrixDims  - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createR2CFftPlan1DZ(const DimensionSizes& inMatrixDims);

    /**
     * @brief   Create cuFFT plan for 1DX Complex-to-Real transform.
     * @details This version doesn't need any scratch place for planning.  All 1D transforms are done in a single
     *          batch. Data has to be transposed and padded according to the cuFFT data layout before the
     *          transform. The FFT is done in-place.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] outMatrixDims - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createC2RFftPlan1DX(const DimensionSizes& outMatrixDims);
    /**
     * @brief   Create cuFFT plan for 1DY Complex-to-Real transform.
     * @details This version doesn't need any scratch place for planning. All 1D transforms are done in a single
     *          batch. The output matrix is padded and transposed to be padded according to the cuFFT data layout.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] outMatrixDims - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createC2RFftPlan1DY(const DimensionSizes& outMatrixDims);
    /**
     * @brief   Create cuFFT plan for 1DZ Complex-to-Real transform.
     * @details This version doesn't need any scratch place for planning. All 1D transforms are done in a single
     *          batch. The output matrix has to be padded and transposed to be padded according to the cuFFT
     *          data layout.
     *
     *          This routine can only be used for both 3D simulations.
     *
     * @param   [in] outMatrixDims - The dimension sizes of the input matrix.
     * @throw   std::runtime_error - If the plan can't be created.
     */
    static void createC2RFftPlan1DZ(const DimensionSizes& outMatrixDims);

    /**
     * @brief Destroy all static plans created by the application.
     * @throw std::runtime_error - If the plan can't be destroyed.
     */
    static void destroyAllPlansAndStaticData();

    /**
     * @brief Compute forward out-of-place ND (2D/3D) Real-to-Complex transform.
     *
     * @param [in] inMatrix      - Input data for the forward FFT.
     * @throw std::runtime_error - If the plan is not valid.
     */
    void computeR2CFftND(RealMatrix& inMatrix);
    /**
     * @brief Compute forward out-of-place ND (2D/3D) Complex-to-Real transform.
     *
     * @param [out] outMatrix    - Output of the inverse FFT.
     * @throw std::runtime_error - If the plan is not valid.
     */
    void computeC2RFftND(RealMatrix& outMatrix);

    /**
     * @brief   Compute forward out-of-place 1DX Real-to-Complex transform.
     * @details This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] inMatrix      - Input data for the forward FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeR2CFft1DX(RealMatrix& inMatrix);
    /**
     * @brief   Compute 1D out-of-place Real-to-Complex transform in the Z dimension.
     * @details Compute forward out-of-place 1DY Real-to-Complex transform. The matrix is first X<->Y transposed
     *          followed by the 1D FFTs. The matrix is left in the transposed format.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [in] inMatrix      - Input data for the forward FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeR2CFft1DY(RealMatrix& inMatrix);
    /**
     * @brief   Compute 1D out-of-place Real-to-Complex transform in the Z dimension.
     * @details Compute forward out-of-place 1DY Real-to-Complex transform. The matrix is first X<->Z transposed
     *          followed by the 1D FFTs. The matrix is left in the transposed format.
     *
     *          This routine can only be used for 3D simulations.
     *
     * @param   [in] inMatrix      - Input data for the forward FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeR2CFft1DZ(RealMatrix& inMatrix);

    /**
     * @brief   Compute inverse out-of-place 1DX Real-to-Complex transform.
     *
     * @details This routine can be used for both 2D and 3D simulations.
     *
     * @param   [out] outMatrix    - Output data for the inverse FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeC2RFft1DX(RealMatrix& outMatrix);
    /**
     * @brief   Compute 1D out-of-place Real-to-Complex transform in the Y dimension.
     * @details Compute inverse out-of-place 1DY Real-to-Complex transform. The matrix is requested to be in the
     *          transposed layout. After the FFT is calculated, an Y<->X transposed follows. The matrix is
     *          returned in the normal layout (z, y, x)  format.
     *
     *          This routine can be used for both 2D and 3D simulations.
     *
     * @param   [out] outMatrix    - Output data for the inverse FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeC2RFft1DY(RealMatrix& outMatrix);
    /**
     * @brief   Compute 1D out-of-place Real-to-Complex transform in the Y dimension.
     * @details Compute forward out-of-place 1DY Real-to-Complex FFT. The matrix is requested to be in the
     *          transposed layout. After the FFT is calculated, an Z<->X transposed follows. The matrix is
     *          returned in the normal layout (z, y, x).
     *
     *          This routine can only be used for 3D simulations.
     *
     * @param   [out] outMatrix    - Output data for the inverse FFT.
     * @throw   std::runtime_error - If the plan is not valid.
     */
    void computeC2RFft1DZ(RealMatrix& outMatrix);

  protected:
    /// cufft plan for the ND Real-to-Complex transform.
    static cufftHandle sR2CFftPlanND;
    /// cufft plan for the ND Complex-to-Real transform.
    static cufftHandle sC2RFftPlanND;

    /// cufft plan for the 1D Real-to-Complex transform in the x dimension.
    static cufftHandle sR2CFftPlan1DX;
    /// cufft plan for the 1D Real-to-Complex transform in the y dimension.
    static cufftHandle sR2CFftPlan1DY;
    /// cufft plan for the 1D Real-to-Complex transform in the z dimension.
    static cufftHandle sR2CFftPlan1DZ;

    /// cufft plan for the 1D Complex-to-Real transform in the x dimension.
    static cufftHandle sC2RFftPlan1DX;
    /// cufft plan for the 1D Complex-to-Real transform in the y dimension.
    static cufftHandle sC2RFftPlan1DY;
    /// cufft plan for the 1D Complex-to-Real transform in the z dimension.
    static cufftHandle sC2RFftPlan1DZ;

  private:
    /**
    * @brief Throw cuda FFT exception.
    * @param [in] cufftError        - CUDA FFT error code.
    * @param [in] transformTypeName - CUDA transform type name.
    * @throw std::runtime_error with message corresponding to the cufft error code.
    */
    static void throwCufftException(const cufftResult  cufftError,
                                    const std::string& transformTypeName);

    /// Error messages for cufft error codes
    static std::map<cufftResult, ErrorMessage> sCufftErrorMessages;
};// CufftComplexMatrix
//----------------------------------------------------------------------------------------------------------------------
#endif /* CUFFT_COMPLEX_MATRIX_H */

/**
 * @file      TransposeCudaKernels.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for CUDA transpose kernels.
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

#ifndef TRANSPOSE_CUDA_KERNELS_CUH
#define TRANSPOSE_CUDA_KERNELS_CUH

/**
 * @namespace TransposeCudaKernels
 * @brief     List of cuda kernels used for matrix transposition.
 * @details   List of cuda kernels used for matrix transposition.
 */
namespace TransposeCudaKernels
{
  /**
   * @enum TransposePadding
   * @brief How the data is padded during matrix transposition.
   */
  enum class TransposePadding
  {
    /// No padding.
    kNone,
    /// Input matrix is padded.
    kInput,
    /// Output matrix is padded.
    kOutput,
    /// Both matrices are padded.
    kInputOutput
  };

  /**
   * @brief Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
   *
   * @tparam      padding      - Which matrices are padded.
   *
   * @param [out] outputMatrix - Output matrix data.
   * @param [in]  inputMatrix  - Input  matrix data.
   * @param [in]  dimSizes     - Dimension sizes of the original matrix.
   */
  template<TransposePadding padding>
  void trasposeReal3DMatrixXY(float*       outputMatrix,
                              const float* inputMatrix,
                              const dim3&  dimSizes);

  /**
   * @brief Transpose a real 3D matrix in the X-Z direction. It is done out-of-place.
   *
   * @tparam      padding      - Which matrices are padded.
   *
   * @param [out] outputMatrix - Output matrix.
   * @param [in]  inputMatrix  - Input  matrix.
   * @param [in]  dimSizes     - Dimension sizes of the original matrix.
   */
    template<TransposePadding padding>
    void trasposeReal3DMatrixXZ(float*       outputMatrix,
                                const float* inputMatrix,
                                const dim3&  dimSizes);
}// end of TransposeCudaKernels
//----------------------------------------------------------------------------------------------------------------------
#endif /* TRANSPOSE_CUDA_KERNELS_CUH */

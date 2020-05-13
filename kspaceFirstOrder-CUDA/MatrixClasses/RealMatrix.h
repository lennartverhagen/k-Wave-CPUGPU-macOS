/**
 * @file      RealMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the class for real matrices.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2011, 10:30 (created) \n
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

#ifndef REAL_MATRIX_H
#define REAL_MATRIX_H

#include <MatrixClasses/BaseFloatMatrix.h>
#include <Utils/DimensionSizes.h>

// Forward declaration
class ComplexMatrix;

/**
 * @class   RealMatrix
 * @brief   The class for real matrices.
 * @details The class for real matrices based on the float datatype on both CPU and GPU side.
 */
class RealMatrix : public BaseFloatMatrix
{
  public:
    /// Default constructor is not allowed.
    RealMatrix() = delete;
    /**
     * @brief Constructor.
     * @param [in] dimensionSizes - Dimension sizes of the matrix.
     */
    RealMatrix(const DimensionSizes& dimensionSizes);
    /// Copy constructor not allowed.
    RealMatrix(const RealMatrix&) = delete;
    /// Destructor.
    virtual ~RealMatrix() override;

    /// Operator = is not allowed.
    RealMatrix& operator=(const RealMatrix&) = delete;

    /**
     * @brief Read matrix from HDF5 file.
     * @param [in] file       - Handle to the HDF5 file.
     * @param [in] matrixName - HDF5 dataset name to read from.
     * @throw ios::failure    - If error occurred.
     */
    virtual void readData(Hdf5File&         file,
                          const MatrixName& matrixName) override;
    /**
     * @brief Write data into HDF5 file.
     * @param [in] file             - Handle to the HDF5 file
     * @param [in] matrixName       - HDF5 dataset name to write to.
     * @param [in] compressionLevel - Compression level for the HDF5 dataset.
     * @throw ios::failure          - If an error occurred.
     */
    virtual void writeData(Hdf5File&         file,
                           const MatrixName& matrixName,
                           const size_t      compressionLevel) override;

    /**
     * @brief   Operator [].
     * @details This operator is not used in the code due to persisting performance issues with vectorization.
     * @param   [in] index - 1D index into the matrix.
     * @return  An element of the matrix.
     */
    inline float&       operator[](const size_t& index)       { return mHostData[index]; };
    /**
     * @brief   Operator [], constant version.
     * @details This operator is not used in the code due to persisting performance issues with vectorization.
     * @param   [in] index - 1D index into the matrix.
     * @return  An element of the matrix.
     */
    inline const float& operator[](const size_t& index) const { return mHostData[index]; };

  protected:

  private:
    /**
     * @brief Initialize dimension sizes.
     * @param [in] dimensionSizes - Dimension sizes of the matrix.
     */
    void initDimensions(const DimensionSizes& dimensionSizes);

    /// Number of elements to get 8MB block of data.
    static constexpr size_t kChunkSize1D8MB   = 2 * 1024 * 1024; //(8MB)
    /// Number of elements to get 1MB block of data.
    static constexpr size_t kChunkSize1D1MB   =      256 * 1024; //(1MB)
    /// Number of elements to get 256KB block of data.
    static constexpr size_t kChunkSize1D128kB =       32 * 1024; //(128KB)
};// end of RealMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* REAL_MATRIX_H */

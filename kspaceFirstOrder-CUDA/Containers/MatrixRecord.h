/**
 * @file      MatrixRecord.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing metadata about matrices stored in the matrix container.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      02 December  2014, 15:44 (created) \n
 *            11 February  2020, 16:10 (revised)
 *
 * @copyright Copyright (C) 2014 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#ifndef MATRIX_RECORD_H
#define MATRIX_RECORD_H

#include <MatrixClasses/BaseMatrix.h>

/**
 * @struct MatrixRecord
 * @brief  A structure storing details about the matrix in the matrix container.
 */
struct MatrixRecord
{
  /**
   * @enum  MatrixType
   * @brief All possible types of the matrix.
   */
  enum class MatrixType
  {
    /// Matrix for real values.
    kReal,
    /// Matrix for complex values.
    kComplex,
    /// Matrix for index values.
    kIndex,
    /// Matrix for cuda fft.
    kCufft
  };

  /// Default constructor.
  MatrixRecord()
    : matrixPtr(nullptr),
      matrixType(MatrixType::kReal),
      dimensionSizes(),
      loadData(false),
      checkpoint(false),
      matrixName()
  {};

  /**
   * @brief Set all values for the record.
   * @param [in] matrixType     - Matrix data type.
   * @param [in] dimensionSizes - Dimension sizes.
   * @param [in] loadData       - Load data from file?
   * @param [in] checkpoint     - Checkpoint this matrix?
   * @param [in] matrixName     - HDF5 matrix name.
   */
  MatrixRecord(const MatrixType     matrixType,
               const DimensionSizes dimensionSizes,
               const bool           loadData,
               const bool           checkpoint,
               const MatrixName&    matrixName)
    : matrixPtr(nullptr),
      matrixType(matrixType),
      dimensionSizes(dimensionSizes),
      loadData(loadData),
      checkpoint(checkpoint),
      matrixName(matrixName)
  {};

  /// Copy constructor.
  MatrixRecord(const MatrixRecord&) = default;
  /// operator =
  MatrixRecord& operator=(const MatrixRecord&) = default;
  // Destructor.
  ~MatrixRecord() = default;

  /// Pointer to the matrix object.
  BaseMatrix*    matrixPtr;
  /// Matrix data type.
  MatrixType     matrixType;
  /// Matrix dimension sizes.
  DimensionSizes dimensionSizes;
  /// Is the matrix content loaded from the HDF5 file?
  bool           loadData;
  /// Is the matrix necessary to be staged in the file when checkpoint is enabled?
  bool           checkpoint;
  /// Matrix name in the HDF5 file.
  MatrixName     matrixName;
};// end of MatrixRecord
//----------------------------------------------------------------------------------------------------------------------

#endif	/* MATRIX_RECORD_H */

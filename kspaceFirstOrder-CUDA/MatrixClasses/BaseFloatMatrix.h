/**
 * @file      BaseFloatMatrix.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the base class for single precisions floating point numbers (floats).
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2011, 12:13 (created) \n
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

#ifndef BASE_FLOAT_MATRIX_H
#define BASE_FLOAT_MATRIX_H

#include <MatrixClasses/BaseMatrix.h>
#include <Utils/DimensionSizes.h>

/**
 * @class   BaseFloatMatrix
 * @brief   Abstract base class for float based matrices defining basic interface. Higher dimensional
 *          matrices stored as 1D arrays, row-major order.
 *
 * @details Abstract base class for float based matrices defining basic interface. Higher dimensional matrices stored
 *          as 1D arrays, row-major order.Implemented both on CPU and GPU side. The I/O is done via HDF5 files.
 */
class BaseFloatMatrix : public BaseMatrix
{
  public:
    /// Default constructor.
    BaseFloatMatrix();
    /// Copy constructor not allowed.
    BaseFloatMatrix(const BaseFloatMatrix&) = delete;
    /// Destructor.
    virtual ~BaseFloatMatrix() override = default;

    /// Operator = not allowed.
    BaseFloatMatrix& operator=(const BaseFloatMatrix&) = delete;

    /**
     * @brief  Get dimension sizes of the matrix.
     * @return Dimension sizes of the matrix.
     */
    virtual const DimensionSizes& getDimensionSizes() const override { return mDimensionSizes; };
    /**
     * @brief  Size of the matrix.
     * @return Number of elements.
     */
    virtual size_t size()                             const override { return mSize; };
    /**
     * @brief  The capacity of the matrix (this may differ from size due to padding, etc.).
     * @return Capacity of the currently allocated storage.
     */
    virtual size_t capacity()                         const override { return mCapacity; };

    /**
     * @brief  Get matrix data stored on the host side (for direct host kernels).
     * @return Pointer to mutable matrix data stored on the host side.
     */
    virtual float*       getHostData()                      { return mHostData; }
    /**
     * @brief  Get matrix data stored on the host side (for direct host kernels), const version.
     * @return Pointer to immutable matrix data stored on the host side.
     */
    virtual const float* getHostData()                const { return mHostData; }
    /**
     * @brief  Get matrix data stored on the device side (for direct device kernels).
     * @return Pointer to immutable matrix data on the device side.
     */
    virtual float*       getDeviceData()                    { return mDeviceData; }
    /**
     * @brief  Get matrix data stored on the device side (for direct device kernels), const version.
     * @return Pointer to immutable matrix data on the device side.
     */
    virtual const float* getDeviceData()              const { return mDeviceData; }

    /// Copy data from host -> device (CPU -> GPU).
    virtual void copyToDevice() override;
    /// Copy data from device -> host (GPU -> CPU).
    virtual void copyFromDevice() override;

    /// Zero all elements of the matrix (NUMA first touch).
    virtual void zeroMatrix();
    /// Zero all elements of the matrix on device.
    virtual void zeroDeviceMatrix();

    /**
     * @brief Calculate matrix = scalar / matrix on the host side.
     * @param [in] scalar - Scalar constant
     */
    virtual void scalarDividedBy(const float scalar);

  protected:
   /**
    * @brief Aligned memory allocation (both on host and device).
    * @throw std::bad_alloc - If there's not enough memory.
    */
    virtual void allocateMemory();
    /// Memory deallocation (both on CPU and GPU).
    virtual void freeMemory();

    /// Dimension sizes.
    DimensionSizes mDimensionSizes;

    /// Total number of used elements.
    size_t mSize;
    /// Total number of allocated elements (in terms of floats).
    size_t mCapacity;

    /// Raw CPU matrix data.
    float* mHostData;
    /// Raw GPU matrix data.
    float* mDeviceData;

  private:

};//end of BaseFloatMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* BASE_FLOAT_MATRIX_H */

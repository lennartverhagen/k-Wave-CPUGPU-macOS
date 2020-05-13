/**
 * @file Types/FftMatrix.h
 *
 * @brief Matrix class with in-place FFT support
 *
 * <!-- GENERATED DOCUMENTATION -->
 * <!-- WARNING: ANY CHANGES IN THE GENERATED BLOCK WILL BE OVERWRITTEN BY THE SCRIPTS -->
 *
 * @author
 * **Jakub Budisky**\n
 * *Faculty of Information Technology*\n
 * *Brno University of Technology*\n
 * ibudisky@fit.vutbr.cz
 *
 * @author
 * **Jiri Jaros**\n
 * *Faculty of Information Technology*\n
 * *Brno University of Technology*\n
 * jarosjir@fit.vutbr.cz
 *
 * @version v1.0.0
 *
 * @date
 * Created: 2017-02-15 09:24\n
 * Last modified: 2020-02-28 08:41
 *
 * @copyright@parblock
 * **Copyright © 2017–2020, SC\@FIT Research Group, Brno University of Technology, Brno, CZ**
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 *
 * @endparblock
 *
 * <!-- END OF GENERATED DOCUMENTATION -->
 **/

#ifndef FFTMATRIX_H
#define FFTMATRIX_H

#include <complex>
#include <memory>
#include <type_traits>

#ifdef USE_FFTWF
  #include <fftw3.h>
#else // Using MKL
  #include <mkl_dfti.h>
#endif

#include <Types/AlignedArray.h>
#include <Types/TypedTuple.h>

/// Type used for 3-dimensional size and position
using Size3D = TypedTuple<std::size_t, 3>;

/**
 * @brief Class encapsulating 3D complex float matrix with FFT routines
 *
 * This class uses AlignedArray as a storage backend and provides a coordinate access and routines to calculate in-place
 * FFT of the content. Two backends for FFT are supported, FFTW and Intel MKL. The complex number are stored in separate
 * arrays and a split FFT interfaces are used.
 */
class FftMatrix
{
  public:
    /**
     * @brief Constructor
     *
     * The content is zero-initialized and FFT plans are created during the object construction.
     *
     * @param[in] size        – Size of the matrix
     * @param[in] threadCount – Number of threads to use for the FFT
     * @throws std::runtime_error if the FFT plan creation fails
     * @throws std::bad_alloc if the space allocation fails
     */
    FftMatrix(const Size3D& size,
              int           threadCount);

    /**
     * @brief Perform in-place forward FFT
     * @throws std::runtime_error if the calculation fails
     */
    void performForwardFft();

    /**
     * @brief Perform in-place backward FFT
     * @throws std::runtime_error if the calculation fails
     */
    void performBackwardFft();

    /**
     * @brief Method to obtain size of the matrix
     * @returns Size of the matrix
     */
    Size3D size() const { return mSize; }

    /**
     * @brief Access a real component of an element of the matrix
     * @param[in] position – Position of the element
     * @returns Reference to the real component of element at the given location
     */
    float& real(const Size3D& position) { return real(position.x(), position.y(), position.z()); }

    /**
     * @brief Access a real component of an element of the matrix
     * @param[in] x – Position of the element in x direction
     * @param[in] y – Position of the element in y direction
     * @param[in] z – Position of the element in z direction
     * @returns Reference to the real component of element at the given location
     */
    float& real(const std::size_t x,
                const std::size_t y,
                const std::size_t z)
    {
      return mRealData[z + mSize.z() * (y + mSize.y() * x)];
    }

    /**
     * @brief Access a real component of an element of the matrix, immutable version
     * @param[in] position – Position of the element
     * @returns Constant reference to the real component of element at the given location
     */
    const float& real(const Size3D& position) const { return real(position.x(), position.y(), position.z()); }

    /**
     * @brief Access a real component of an element of the matrix, immutable version
     * @param[in] x – Position of the element in x direction
     * @param[in] y – Position of the element in y direction
     * @param[in] z – Position of the element in z direction
     * @returns Constant reference to the real component of element at the given location
     */
    const float& real(const std::size_t x,
                      const std::size_t y,
                      const std::size_t z) const
    {
      return mRealData[z + mSize.z() * (y + mSize.y() * x)];
    }

    /**
     * @brief Access an imaginary component of an element of the matrix
     * @param[in] position – Position of the element
     * @returns Reference to the imaginary component of element at the given location
     */
    float& imag(const Size3D& position) { return imag(position.x(), position.y(), position.z()); }

    /**
     * @brief Access an imaginary component of an element of the matrix
     * @param[in] x – Position of the element in x direction
     * @param[in] y – Position of the element in y direction
     * @param[in] z – Position of the element in z direction
     * @returns Reference to the imaginary component of element at the given location
     */
    float& imag(const std::size_t x,
                const std::size_t y,
                const std::size_t z)
    {
      return mImagData[z + mSize.z() * (y + mSize.y() * x)];
    }

    /**
     * @brief Access an imaginary component of an element of the matrix, immutable version
     * @param[in] position – Position of the element
     * @returns Constant reference to the imaginary component of element at the given location
     */
    const float& imag(const Size3D& position) const { return imag(position.x(), position.y(), position.z()); }

    /**
     * @brief Access an imaginary component of an element of the matrix, immutable version
     * @param[in] x – Position of the element in x direction
     * @param[in] y – Position of the element in y direction
     * @param[in] z – Position of the element in z direction
     * @returns Constant reference to the imaginary component of element at the given location
     */
    const float& imag(const std::size_t x,
                      const std::size_t y,
                      const std::size_t z) const
    {
      return mImagData[z + mSize.z() * (y + mSize.y() * x)];
    }

    /// Method to access the underlying real data
    float* realData()             { return mRealData.begin(); }
    /// Method to access the underlying real data, immutable version
    const float* realData() const { return mRealData.cbegin(); }
    /// Method to access the underlying imaginary data
    float* imagData()             { return mImagData.begin(); }
    /// Method to access the underlying imaginary data, immutable version
    const float* imagData() const { return mImagData.cbegin(); }

  private:
    /// Size of the matrix
    const Size3D            mSize;
    /// Internal storage
    AlignedArray<float, 64> mRealData;
    AlignedArray<float, 64> mImagData;

    // Backend-dependent members
#ifdef USE_FFTWF
    using FftPlan = std::remove_pointer<fftwf_plan>::type;

    /// Helper class to free FFT plans, FFTWF version
    class FftPlanDeleter
    {
      public:
        void operator()(FftPlan* plan) { fftwf_destroy_plan(plan); }
    };

    /// Initialization class for FFTWF
    class FftwfInitializer
    {
      public:
        /// Constructor
        FftwfInitializer()
        {
          if (!fftwf_init_threads())
          {
            throw std::runtime_error("Failed to initialize FFTWF threads");
          }
        }
        /// Destructor
        ~FftwfInitializer() { fftwf_cleanup_threads(); }
    };

    /// Static object to initialize and free FFTWF internal objects
    const static FftwfInitializer            kInitializer;
    /// Forward FFT plan
    std::unique_ptr<FftPlan, FftPlanDeleter> mFftForwardPlan;
    /// Backward FFT plan
    std::unique_ptr<FftPlan, FftPlanDeleter> mFftBackwardPlan;

#else // Using MKL
    using FftPlan = std::remove_pointer<DFTI_DESCRIPTOR_HANDLE>::type;

    /// Helper class to free FFT plans, MKL version
    class FftPlanDeleter
    {
      public:
        void operator()(FftPlan* plan) { DftiFreeDescriptor(&plan); }
    };

    /// Single FFT plan for MKL
    std::unique_ptr<FftPlan, FftPlanDeleter> mFftPlan;
#endif

};// end of FftMatrix
//----------------------------------------------------------------------------------------------------------------------

#endif /* FFTMATRIX_H */

/**
 * @file Types/FftMatrixFftwf.cpp
 *
 * @brief Matrix class with in-place FFT FFTW support
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

#include <Types/FftMatrix.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// Static variable ensuring the call to fftwf_init_threads() and fftwf_cleanup_threads()
const FftMatrix::FftwfInitializer FftMatrix::kInitializer;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor.
 */
FftMatrix::FftMatrix(const Size3D& size,
                     int           threadCount)
  : mSize(size),
    mRealData(size.product(), 0.0f),
    mImagData(size.product(), 0.0f)
{
  // set the number of threads
  fftwf_plan_with_nthreads(threadCount);

  using DimType = decltype(fftw_iodim64::n);
  static_assert(sizeof(Size3D::value_type) == sizeof(DimType),
                "Size of the type used for the FFTWF interface description does not match the size of the type for "
                "specifying dimensionality of FftMatrix and this implementation is malformed.");

  // we assume DimType being signed, but for transform to work correctly we would need the element count be less than
  // the limit anyway, if it overflows we expect the code to yield wrong results but the memory should be allocated

  // clang-format off
  // prepare the descriptors for planning
  const std::array<fftwf_iodim64, 3> dims = {
      //            size               input stride                  output stride
      fftwf_iodim64{DimType(size.x()), DimType(size.z() * size.y()), DimType(size.z() * size.y())},
      fftwf_iodim64{DimType(size.y()), DimType(size.z()),            DimType(size.z())           },
      fftwf_iodim64{DimType(size.z()), 1,                            1                           }
  };
  // clang-format on

  // create FFT plans
  mFftForwardPlan.reset(fftwf_plan_guru64_split_dft(3, dims.data(), 0, nullptr, mRealData.begin(), mImagData.begin(),
                                                    mRealData.begin(), mImagData.begin(), FFTW_ESTIMATE));
  if (!mFftForwardPlan)
  {
    throw std::runtime_error("Failed to create forward FFTWF plan");
  }

  mFftBackwardPlan.reset(fftwf_plan_guru64_split_dft(3, dims.data(), 0, nullptr, mImagData.begin(), mRealData.begin(),
                                                     mImagData.begin(), mRealData.begin(), FFTW_ESTIMATE));
  if (!mFftBackwardPlan)
  {
    throw std::runtime_error("Failed to create backward FFTWF plan");
  }
}// end of FftMatrix::FftMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Perform in-place FFT
 */
void FftMatrix::performForwardFft()
{
  fftwf_execute(mFftForwardPlan.get());
}// end of FftMatrix::performForwardFft
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Perform in-place backward FFT
 */
void FftMatrix::performBackwardFft()
{
  fftwf_execute(mFftBackwardPlan.get());
}// end of FftMatrix::performBackwardFft
//----------------------------------------------------------------------------------------------------------------------

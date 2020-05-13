/**
 * @file Types/FftMatrixMkl.cpp
 *
 * @brief Matrix class with in-place FFT MKL support
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
 * Created: 2017-06-28 11:25\n
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
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor
 */
FftMatrix::FftMatrix(const Size3D& size,
                     int           threadCount)
  : mSize(size),
    mRealData(size.product(), 0.0f),
    mImagData(size.product(), 0.0f)
{
  // Perform type check
  static_assert(sizeof(Size3D::value_type) == sizeof(MKL_LONG),
                "MKL_LONG has a different size than std::size_t and this implementation is malformed");
  // check for the limits is omitted due to addressable memory constraint (casting signed MKL_LONG -> unsigned size_t)

  // Create the plan into a temporary handle
  FftPlan* handle;
  MKL_LONG status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 3, size.data());
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }
  mFftPlan.reset(handle);

  // Set the complex storage for the split DFT
  status = DftiSetValue(handle, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }

  // Set the maximum number of threads to use
  status = DftiSetValue(handle, DFTI_THREAD_LIMIT, static_cast<MKL_LONG>(threadCount));
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }

  // Commit the descriptor
  status = DftiCommitDescriptor(handle);
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }
}// end of FftMatrix::FftMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Perform in-place forward FFT
 */
void FftMatrix::performForwardFft()
{
  MKL_LONG status = DftiComputeForward(mFftPlan.get(), mRealData.begin(), mImagData.begin());
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }
}// end of FftMatrix::performForwardFft
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Perform in-place backward FFT
 */
void FftMatrix::performBackwardFft()
{
  MKL_LONG status = DftiComputeBackward(mFftPlan.get(), mRealData.begin(), mImagData.begin());
  if (status)
  {
    throw std::runtime_error(DftiErrorMessage(status));
  }
}// end of FftMatrix::performBackwardFft
//----------------------------------------------------------------------------------------------------------------------

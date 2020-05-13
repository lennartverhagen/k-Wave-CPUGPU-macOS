/**
 * @file Kernels/Kernels.h
 *
 * @brief Computational kernels
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

#ifndef KERNELS_H
#define KERNELS_H

#include <Types/FftMatrix.h>
#include <Types/Parameters.h>

namespace Kernels
{
  /**
   * @brief Kernel transforming the input into a complex form
   *
   * @param[in]     params – Calculation parameters
   * @param[in,out] matrix – Input amplitude and phase, complex format on return
   */
  void preprocess(const Parameters& params,
                  FftMatrix&        matrix);

  /**
   * @brief Kernel calculating pressure field in a given time using Green's function
   *
   * This kernel works with spectrum representation, calculating acoustic field for homogeneous non-absorbing media
   * using a Green's function for the linear wave equation.
   *
   * @param[in]     params – Calculation parameters
   * @param[in,out] matrix – Spectrum of the pressure field at t = 0, at t = t1 on return
   */
#ifndef DEBUG_DATASET
  void advanceWaves(const Parameters& params,
                    FftMatrix&        matrix);
#else
  void advanceWaves(const Parameters& params,
                    FftMatrix&        matrix,
                    FftMatrix&        w,
                    FftMatrix&        p);
#endif

  /**
   * @brief Kernel normalizing the result after doing FFT
   *
   * @warning You should **not** be using this before a call to Kernels::recovery
   *
   * @param[in]     params – Calculation parameters
   * @param[in,out] matrix – Matrix to normalize
   */
  void normalize(const Parameters& params,
                 FftMatrix&        matrix);

  /**
   * @brief Kernel recovering amplitude and phase from complex pressure
   *
   * @warning Not normalizing inverse FFT (thus inputs) expected, normalization is done as a part of the kernel
   *
   * @param[in]     params – Calculation parameters
   * @param[in,out] matrix – Complex pressure input, amplitude in real component and phase in imaginary component on
   *                         return
   */
  void recovery(const Parameters& params,
                FftMatrix&        matrix);
}// end of Kernels
//----------------------------------------------------------------------------------------------------------------------

#endif /* KERNELS_H */

/**
 * @file Kernels/Kernels.cpp
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

#include <Kernels/Kernels.h>

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Kernel transforming the input into a complex form
 */
void Kernels::preprocess(const Parameters& params,
                         FftMatrix&        matrix)
{
  const std::size_t sizeX = params.size.x();
  const std::size_t sizeY = params.size.y();
  const std::size_t sizeZ = params.size.z();

  if (params.phaseIsScalar)
  {
    const std::complex<float> factor(std::cos(params.phase), std::sin(params.phase));

    #pragma omp parallel for schedule(static)
    for (std::size_t x = 0; x < sizeX; ++x)
    {
      for (std::size_t y = 0; y < sizeY; ++y)
      {
        #pragma omp simd
        for (std::size_t z = 0; z < sizeZ; ++z)
        {
          const auto value     = matrix.real(x, y, z) * factor;
          matrix.real(x, y, z) = value.real();
          matrix.imag(x, y, z) = value.imag();
        }
      }
    }// end of outer for
  }
  else
  {
    #pragma omp parallel for schedule(static)
    for (std::size_t x = 0; x < sizeX; ++x)
    {
      for (std::size_t y = 0; y < sizeY; ++y)
      {
        #pragma omp simd
        for (std::size_t z = 0; z < sizeZ; ++z)
        {
          const auto value = matrix.real(x, y, z) *
                             std::complex<float>(std::cos(matrix.imag(x, y, z)), std::sin(matrix.imag(x, y, z)));
          matrix.real(x, y, z) = value.real();
          matrix.imag(x, y, z) = value.imag();
        }
      }
    }// end of outer for
  }// end of if (params.phaseIsScalar)
}// end of Kernels::preprocess
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Kernel calculating pressure field in a given time using Green's function
 */
#ifndef DEBUG_DATASET
void Kernels::advanceWaves(const Parameters& params,
                           FftMatrix&        matrix)
#else
void Kernels::advanceWaves(const Parameters& params,
                           FftMatrix&        matrix,
                           FftMatrix&        w,
                           FftMatrix&        p)
#endif
{
  // divide the value by two and floor it
  auto div2floor =[](const std::size_t number) -> std::size_t
  {
    return number >> 1;
  };// end of div2floor

  const std::size_t sizeX = params.extended.x();
  const std::size_t sizeY = params.extended.y();
  const std::size_t sizeZ = params.extended.z();

  // prepare variables for wave number calculation
  const double kXScale = 2.0 * M_PI / (params.dx * sizeX);
  const double kYScale = 2.0 * M_PI / (params.dx * sizeY);
  const double kZScale = 2.0 * M_PI / (params.dx * sizeZ);

  // prepare wave numbers threshold (max(abs(k)) * eps * 10)
  const double maxKSquared = div2floor(sizeX) * div2floor(sizeX) * kXScale * kXScale +
                             div2floor(sizeY) * div2floor(sizeY) * kYScale * kYScale +
                             div2floor(sizeZ) * div2floor(sizeZ) * kZScale * kZScale;
  const double kThreshold = 10.0 * std::numeric_limits<double>::epsilon() * std::sqrt(maxKSquared);

  // frequently used variables
  const double c0  = params.c0;
  const double w0  = params.w0;
  const double t   = params.t;
  const double c02 = c0 * c0;
  const double c04 = c02 * c02;
  const double w02 = w0 * w0;
  const double w03 = w0 * w02;
  const double w04 = w02 * w02;
  const std::complex<double> i{0, 1};
  const std::complex<double> eiw0t = std::exp(i * t * w0);

  // prepare all the factors for the propagator to limit the calculations inside the loop
  const double c0t1 = c0 * t;
  const double c0t2 = c0 * (t - 2.0 * M_PI / w0);

  const std::complex<double> nomK0factor    = -18.0 * i * w04 * w0 * eiw0t;
  const std::complex<double> nomK2factor    = 80.0 * i * c02 * w03 * eiw0t;
  const std::complex<double> nomK4factor    = -32.0 * i * c04 * w0 * eiw0t;
  const std::complex<double> nomK0cosFactor = -3.0 * i * w04 * w0;
  const double nomK1sinFactor               = 11.0 * c0 * w04;
  const std::complex<double> nomK2cosFactor = -12.0 * i * c02 * w03;
  const double nomK3sinFactor               = 4.0 * c02 * c0 * w02;

  const double denK0factor = 18.0 * w04 * w02;
  const double denK2factor = -98.0 * c02 * w04;
  const double denK4factor = 112.0 * c04 * w02;
  const double denK6factor = -32.0 * c04 * c02;

  // prepare the alternative propagators
  const std::complex<double> limKw0c0 = (15.0 * std::exp(2.0 * i * w0 * t) * (2.0 * w0 * t - i - 2.0 * M_PI) - i) /
                                        (60.0 * w0 * eiw0t);
  const std::complex<double> limKw02c0  = -(16.0 * i * eiw0t + 3.0 * M_PI * std::exp(i * w0 * t / 2.0)) / (12.0 * w0);
  const std::complex<double> limK3w02c0 = (16.0 * i * eiw0t - 5.0 * M_PI * std::exp(3.0 * i * w0 * t / 2.0)) /
                                          (20.0 * w0);
  const std::complex<double> limK0 = -i * (1.0 + 3.0 * eiw0t) / (3.0 * w0);

  // process all the elements in the loop, calculating the wave numbers as needed
  using ssize_t = std::make_signed<std::size_t>::type;

  // main loop
  #pragma omp parallel for schedule(static)
  for (std::size_t x = 0; x < sizeX; ++x)
  {
    const double xDiff = static_cast<ssize_t>((x + div2floor(sizeX)) % sizeX - div2floor(sizeX)) * kXScale;
    const double xPart = xDiff * xDiff;

    for (std::size_t y = 0; y < sizeY; ++y)
    {
      const double yDiff  = static_cast<ssize_t>((y + div2floor(sizeY)) % sizeY - div2floor(sizeY)) * kYScale;
      const double xyPart = yDiff * yDiff + xPart;

      #pragma omp simd
      for (std::size_t z = 0; z < sizeZ; ++z)
      {
        const double zDiff = static_cast<ssize_t>((z + div2floor(sizeZ)) % sizeZ - div2floor(sizeZ)) * kZScale;
        const double k     = std::sqrt(zDiff * zDiff + xyPart);

        const double k2 = k * k;
        const double k4 = k2 * k2;

        // calculate additional expressions
        const double cosTerm = std::cos(k * c0t1) + std::cos(k * c0t2);
        const double sinTerm = std::sin(k * c0t1) + std::sin(k * c0t2);

        std::complex<double> propagator = nomK0factor + nomK0cosFactor * cosTerm + k * nomK1sinFactor * sinTerm +
                                          k2 * (nomK2factor + nomK2cosFactor * cosTerm) +
                                          k2 * k * nomK3sinFactor * sinTerm + k4 * nomK4factor;
        propagator /= denK0factor + k2 * denK2factor + k4 * denK4factor + k4 * k2 * denK6factor;

        propagator = (std::abs(k - w0 / c0) < kThreshold) ? limKw0c0 : propagator;
        propagator = (std::abs(k - w0 / (2.0 * c0)) < kThreshold) ? limKw02c0 : propagator;
        propagator = (std::abs(k - (3.0 * w0) / (2.0 * c0)) < kThreshold) ? limK3w02c0 : propagator;
        propagator = (k == 0.0) ? limK0 : propagator;

        // multiply
        const auto value     = propagator * std::complex<double>(matrix.real(x, y, z), matrix.imag(x, y, z));
        matrix.real(x, y, z) = value.real();
        matrix.imag(x, y, z) = value.imag();

#ifdef DEBUG_DATASET
        p.real(x, y, z) = propagator.real();
        p.imag(x, y, z) = propagator.imag();
        w.real(x, y, z) = k;
#endif
      }
    }
  }// end of outer for
}// end of Kernels::advanceWaves
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Kernel normalizing the result after doing FFT
 */
void Kernels::normalize(const Parameters& params,
                        FftMatrix&        matrix)
{
  // k-Wave normalizing factor
  const float kwaveNormalization = 2.0f * params.c0 / params.dx;
  // FFT normalization
  const float fftNormalization = 1.0f / (params.extended.x() * params.extended.y() * params.extended.z());
  // resulting factor
  const float factor = kwaveNormalization * fftNormalization;

  const std::size_t sizeX = params.size.x();
  const std::size_t sizeY = params.size.y();
  const std::size_t sizeZ = params.size.z();

  #pragma omp parallel for schedule(static)
  for (std::size_t x = 0; x < sizeX; ++x)
  {
    for (std::size_t y = 0; y < sizeY; ++y)
    {
      #pragma omp simd
      for (std::size_t z = 0; z < sizeZ; ++z)
      {
        matrix.real(x, y, z) *= factor;
        matrix.imag(x, y, z) *= factor;
      }
    }
  }// end of outer for
}// end of Kernels::normalize
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Kernel recovering amplitude and phase from complex pressure
 */
void Kernels::recovery(const Parameters& params,
                       FftMatrix&        matrix)
{
  // k-Wave normalizing factor
  const float kwaveNormalization = 2.0f * params.c0 / params.dx;
  // FFT normalization
  const float fftNormalization = 1.0f / (params.extended.x() * params.extended.y() * params.extended.z());
  // resulting factor
  const float factor = kwaveNormalization * fftNormalization;

  const std::size_t sizeX = params.size.x();
  const std::size_t sizeY = params.size.y();
  const std::size_t sizeZ = params.size.z();

  #pragma omp parallel for schedule(static)
  for (std::size_t x = 0; x < sizeX; ++x)
  {
    for (std::size_t y = 0; y < sizeY; ++y)
    {
      #pragma omp simd
      for (std::size_t z = 0; z < sizeZ; ++z)
      {
        const auto value     = std::complex<float>(matrix.real(x, y, z), matrix.imag(x, y, z)) * factor;
        matrix.real(x, y, z) = std::abs(value);
        matrix.imag(x, y, z) = std::arg(value);
      }
    }
  }// end of outer for
}// end of Kernels::recovery
//----------------------------------------------------------------------------------------------------------------------

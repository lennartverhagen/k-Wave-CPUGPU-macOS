/**
 * @file      CudaDeviceConstants.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for the class for storing constants residing in CUDA constant memory.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      17 February  2016, 10:53 (created) \n
 *            11 February  2020, 16:21 (revised)
 *
 * @copyright Copyright (C) 2016 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#ifndef CUDA_DEVICE_CONSTANTS_H
#define CUDA_DEVICE_CONSTANTS_H

#include <Parameters/Parameters.h>

/**
  * @struct CudaDeviceConstants
  * @brief  Structure for CUDA parameters to be placed in constant memory. Only 32b values are used, since CUDA does
  *         not allow to allocate more than 2^32 elements and dim3 datatype is based on unsigned int.
  */
struct CudaDeviceConstants
{
  /// Upload device constants into GPU memory.
  __host__ void copyToDevice();

  /// Is the simulation 2D or 3D.
  Parameters::SimulationDimension simulationDimension;
  /// Size of x dimension.
  unsigned int nx;
  /// Size of y dimension.
  unsigned int ny;
  /// Size of z dimension.
  unsigned int nz;
  /// Total number of elements.
  unsigned int nElements;
  /// Size of complex x dimension.
  unsigned int nxComplex;
  /// Size of complex y dimension.
  unsigned int nyComplex;
  /// Size of complex z dimension.
  unsigned int nzComplex;
  /// Complex number of elements.
  unsigned int nElementsComplex;
  /// Normalization constant for 3D FFT.
  float fftDivider;
  /// Normalization constant for 1D FFT over x.
  float fftDividerX;
  /// Normalization constant for 1D FFT over y.
  float fftDividerY;
  /// Normalization constant for 1D FFT over z.
  float fftDividerZ;

  /// dt
  float dt;
  /// 2.0 * dt
  float dtBy2;
  /// c^2
  float c2;

  /// rho0 in homogeneous case.
  float rho0;
  /// dt * rho0 in homogeneous case.
  float dtRho0;
  /// dt / rho0Sgx in homogeneous case.
  float dtRho0Sgx;
  /// dt / rho0Sgy in homogeneous case.
  float dtRho0Sgy;
  /// dt / rho0Sgz in homogeneous case.
  float dtRho0Sgz;

  /// B/A value for homogeneous case.
  float bOnA;

  /// AbsorbTau value for homogeneous case.
  float absorbTau;
  /// AbsorbEta value for homogeneous case.
  float absorbEta;

  /// Size of the velocity source.
  unsigned int velocitySourceSize;
  /// Velocity source mode.
  Parameters::SourceMode velocitySourceMode;
  /// Velocity source many.
  unsigned int velocitySourceMany;

  /// Size of the pressure source mask.
  unsigned int presureSourceSize;
  /// Pressure source mode.
  Parameters::SourceMode presureSourceMode;
  /// Pressure source many.
  unsigned int presureSourceMany;
}; // end of CudaDeviceConstants
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUDA_DEVICE_CONSTANTS_H */

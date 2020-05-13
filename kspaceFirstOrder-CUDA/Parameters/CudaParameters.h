/**
 * @file      CudaParameters.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for the class for setting CUDA kernel parameters.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 November  2015, 16:49 (created) \n
 *            11 February  2020, 16:21 (revised)
 *
 * @copyright Copyright (C) 2015 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#ifndef CUDA_PARAMETERS_H
#define CUDA_PARAMETERS_H

#include <cuda_runtime.h>

#include <Utils/DimensionSizes.h>

/**
 * @class   CudaParameters
 * @brief   Class responsible for CUDA runtime setup.
 * @details Class responsible for selecting a CUDA device, block and grid dimensions, etc. \n
 *          The class can only be constructed from inside Parameters and there mustn't be more
 *          than 1 instance in the code.
 */
class CudaParameters
{
  public:
    /// Only Parameters can create this class.
    friend class Parameters;

    /// Copy constructor not allowed.
    CudaParameters(const CudaParameters&) = delete;
    /// Destructor.
    ~CudaParameters() = default;

    /// Operator = not allowed.
    CudaParameters& operator=(const CudaParameters&) = delete;

    /**
     * @brief  Get index of the device being used.
     * @return Index of selected CUDA device.
     */
    int getDeviceIdx()                 const { return mDeviceIdx; }

    /**
     * @brief  Get number of threads for 1D block used by kSpaceSolver.
     * @return Number of threads per block.
     */
    int getSolverBlockSize1D()         const { return mSolverBlockSize1D; }
    /**
     * @brief  Get number of block for 1D grid used by kSpaceSolver.
     * @return Number of blocks per grid.
     */
    int getSolverGridSize1D()          const { return mSolverGridSize1D; }

    /**
     * @brief  Get block size for the transposition kernels.
     * @return Number of threads per block.
     */
    dim3 getSolverTransposeBlockSize() const { return mSolverTransposeBlockSize; }
    /**
     * @brief  Get grid size for the transposition kernels.
     * @return Number of blocks per grid.
     */
    dim3 getSolverTransposeGirdSize()  const { return mSolverTransposeGirdSize; }

    /**
     * @brief  Get number of threads for the 1D data sampling kernels.
     * @return Number of threads per block.
     */
    int getSamplerBlockSize1D()        const { return mSamplerBlockSize1D; }
    /**
     * @brief  Get Number of blocks for the 1D data sampling kernels.
     * @return Number of blocks per grid.
     */
    int getSamplerGridSize1D()         const { return mSamplerGridSize1D; }

    /**
     * @brief  Get the name of the device being used.
     * @return Name of the GPU card being used, e.g., GeForce GTX 980 or "N/A".
     */
    std::string getDeviceName()        const;

    /**
     * @brief Select cuda device for execution.
     * @param [in] deviceIdx     - Device to acquire, default is the first free.
     *
     * @throw std::runtime_error - If there is no free CUDA devices.
     * @throw std::runtime_error - If there is no device of such and deviceIdx.
     * @throw std::runtime_error - If the GPU chosen is not supported (i.e., the code was not compiled for its
     *                             architecture).
     */
    void selectDevice(const int deviceIdx = kDefaultDeviceIdx);

    /// Set kernel configurations based on the simulation parameters.
    void setKernelConfiguration();

    /// Upload useful simulation constants into device constant memory.
    void setUpDeviceConstants()                 const;

    /**
     * @brief  Return properties of the GPU being used.
     * @return Structure holding the properties of GPU being used.
     */
    const cudaDeviceProp& getDeviceProperties() const { return mDeviceProperties; };

    /// Default device Index - no default GPU.
    static constexpr int kDefaultDeviceIdx = -1;

  protected:

  private:
    /// Default constructor - only friend class can create an instance.
    CudaParameters();

    /**
     * @brief Check whether the CUDA driver version installed is sufficient for the code.
     * @throw std::runtime_error if the CUDA driver is too old.
     */
    void checkCudaVersion();

    /**
     * @brief  Check whether the code was compiled for a given SM model.
     * @return true - if we can run the code, the minimum SM model is 3.0 (Kepler).
     */
    bool checkCudaCodeVersion();

    /// Undefined block or grid size.
    static constexpr int kUndefinedSize = -1;

    /// Index of the device the code is being run on.
    int  mDeviceIdx;

    /// Number of threads for 1D block used by kSpaceSolver.
    int  mSolverBlockSize1D;
    /// Number of block for 1D grid used by kSpaceSolver.
    int  mSolverGridSize1D;

    /// Block size for the transposition kernels.
    dim3 mSolverTransposeBlockSize;
    /// Grid size for the transposition kernels.
    dim3 mSolverTransposeGirdSize;

    /// Number of threads for the 1D data sampling kernels.
    int  mSamplerBlockSize1D;
    /// Number of blocks for the 1D data sampling kernels.
    int  mSamplerGridSize1D;

    /// Device properties of the selected GPU.
    cudaDeviceProp mDeviceProperties;
};// end of CudaParameters
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUDA_PARAMETERS_H */

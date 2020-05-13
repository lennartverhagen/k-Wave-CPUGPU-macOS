/**
 * @file      CuboidOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of classes responsible for storing output quantities based
 *            on the cuboid sensor mask into the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      13 February  2015, 12:51 (created) \n
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

#include <OutputStreams/CuboidOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset, SourceMatrix, SensorMask and reduction operator together.
 */
CuboidOutputStream::CuboidOutputStream(Hdf5File&            file,
                                       const MatrixName&    groupName,
                                       const RealMatrix&    sourceMatrix,
                                       const IndexMatrix&   sensorMask,
                                       const ReduceOperator reduceOp)
  : BaseOutputStream(file, groupName, sourceMatrix, reduceOp),
    mSensorMask(sensorMask),
    mGroup(H5I_BADID),
    mSampledTimeStep(0),
    mEventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&mEventSamplingFinished));
}// end of CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
CuboidOutputStream::~CuboidOutputStream()
{
  // Destroy sampling event
  cudaCheckErrors(cudaEventDestroy(mEventSamplingFinished));
  // Close the stream
  close();
  // Free memory
  freeMemory();
}// end ~CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream and allocate data for it. It also creates a HDF5 group with particular datasets
 * (one per cuboid).
 */
void CuboidOutputStream::create()
{
  // Create the HDF5 group and open it
  mGroup = mFile.createGroup(mFile.getRootGroup(), mRootObjectName);

  // Create all datasets (sizes, chunks, and attributes)
  const size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);

  size_t actualPositionInBuffer = 0;

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;

    cuboidInfo.cuboidIdx = createCuboidDataset(cuboidIdx);
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    mCuboidsInfo.push_back(cuboidInfo);

    actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                               mSensorMask.getTopLeftCorner(cuboidIdx)
                              ).nElements();
  }

  // We're at the beginning
  mSampledTimeStep = 0;

  // Create the memory buffer if necessary and set starting address
  mBufferSize = mSensorMask.getSizeOfAllCuboids();

  // Allocate memory
  allocateMemory();
}// end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart and reload data.
 */
void CuboidOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  mSampledTimeStep = 0;
  if (mReduceOp == ReduceOperator::kNone) // Set correct sampled time step for raw data series
  {
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex())
                          ? 0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }

  // Create a memory buffer if necessary and set starting address
  mBufferSize = mSensorMask.getSizeOfAllCuboids();

  // Allocate memory if needed
  allocateMemory();

  // Open all datasets (sizes, chunks, and attributes)
  const size_t nCuboids = mSensorMask.getDimensionSizes().ny;
  mCuboidsInfo.reserve(nCuboids);

  size_t actualPositionInBuffer = 0;

  // Open the HDF5 group
  mGroup = mFile.openGroup(mFile.getRootGroup(), mRootObjectName);

  for (size_t cuboidIdx = 0; cuboidIdx < nCuboids; cuboidIdx++)
  {
    CuboidInfo cuboidInfo;

    // Indexed from 1
    const std::string datasetName = std::to_string(cuboidIdx + 1);

    // Open the dataset
    cuboidInfo.cuboidIdx = mFile.openDataset(mGroup, datasetName);
    cuboidInfo.startingPossitionInBuffer = actualPositionInBuffer;
    mCuboidsInfo.push_back(cuboidInfo);

    // Read only if there is anything to read
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      if (mReduceOp != ReduceOperator::kNone)
      { // Reload data
        DimensionSizes cuboidSize((mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).nx,
                                  (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).ny,
                                  (mSensorMask.getBottomRightCorner(cuboidIdx) -
                                   mSensorMask.getTopLeftCorner(cuboidIdx)).nz);

        mFile.readCompleteDataset(mGroup,
                                  datasetName,
                                  cuboidSize,
                                  mHostBuffer + actualPositionInBuffer);
      }
    }
    // Move the pointer for the next cuboid beginning (this inits the locations)
    actualPositionInBuffer += (mSensorMask.getBottomRightCorner(cuboidIdx) -
                               mSensorMask.getTopLeftCorner(cuboidIdx)).nElements();
  }

  // Copy data over to the GPU only if there is anything to read
  if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
  {
    copyToDevice();
  }
}// end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer an flush to the disk unless a reduction operator is applied.
 */
void CuboidOutputStream::sample()
{
  size_t cuboidInBufferStart = 0;

  // Dimension sizes of the matrix being sampled
  const dim3 dimSizes (static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().nx),
                       static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().ny),
                       static_cast<unsigned int>(mSourceMatrix.getDimensionSizes().nz));

  // Run over all cuboids - this is not a good solution as we need to run a distinct kernel for a cuboid
  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    // Copy down dim sizes
    const dim3 topLeftCorner(static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).nx),
                             static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).ny),
                             static_cast<unsigned int>(mSensorMask.getTopLeftCorner(cuboidIdx).nz));
    const dim3 bottomRightCorner(static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).nx),
                                 static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).ny),
                                 static_cast<unsigned int>(mSensorMask.getBottomRightCorner(cuboidIdx).nz));

    // Get number of samples within the cuboid
    const size_t nSamples = (mSensorMask.getBottomRightCorner(cuboidIdx) -
                             mSensorMask.getTopLeftCorner(cuboidIdx)
                            ).nElements();

    switch (mReduceOp)
    {
      case ReduceOperator::kNone:
      {
        // Kernel to sample raw quantities inside one cuboid
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kNone>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }

      case ReduceOperator::kRms:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kRms>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }

      case ReduceOperator::kMax:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMax>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }

      case ReduceOperator::kMin:
      {
        OutputStreamsCudaKernels::sampleCuboid<ReduceOperator::kMin>
                                              (mDeviceBuffer + cuboidInBufferStart,
                                               mSourceMatrix.getDeviceData(),
                                               topLeftCorner,
                                               bottomRightCorner,
                                               dimSizes,
                                               nSamples);
        break;
      }
    }

    cuboidInBufferStart += nSamples;
  }

  if (mReduceOp == ReduceOperator::kNone)
  {
    // Record an event when the data has been copied over.
    cudaCheckErrors(cudaEventRecord(mEventSamplingFinished));
  }
}// end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush data for the timestep. Only applicable on RAW data series.
 */
void CuboidOutputStream::flushRaw()
{
  if (mReduceOp == ReduceOperator::kNone)
  {
    // Make sure the data has been copied from the GPU
    cudaEventSynchronize(mEventSamplingFinished);

    // Only raw time series are flushed down to the disk every time step
    flushBufferToFile();
  }
}// end of flushRaw
//----------------------------------------------------------------------------------------------------------------------

/*
 * Apply post-processing on the buffer and flush it to the file.
 */
void CuboidOutputStream::postProcess()
{
  // Run inherited method
  BaseOutputStream::postProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone)
  {
    // Copy data from GPU matrix
    copyFromDevice();

    flushBufferToFile();
  }
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream.
 */
void CuboidOutputStream::checkpoint()
{
  // Raw data has already been flushed, others have to be flushed here
  if (mReduceOp != ReduceOperator::kNone)
  {
    // Copy data from the device
    copyFromDevice();
    // Flush to disk
    flushBufferToFile();
  }
}// end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream (apply post-processing if necessary, flush data, close datasets and the group).
 */
void CuboidOutputStream::close()
{
  // The group is still open
  if (mGroup != H5I_BADID)
  {
    // Close all datasets and the group
    for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
    {
      mFile.closeDataset(mCuboidsInfo[cuboidIdx].cuboidIdx);
    }
    mCuboidsInfo.clear();

    mFile.closeGroup(mGroup);
    mGroup = H5I_BADID;
  }// if opened
}// end of close
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create a new dataset for a given cuboid specified by index (order).
 */
hid_t CuboidOutputStream::createCuboidDataset(const size_t cuboidIdx)
{
  const Parameters& params = Parameters::getInstance();

  // If time series then Number of steps else 1
  const size_t nSampledTimeSteps = (mReduceOp == ReduceOperator::kNone)
                                      ? params.getNt() - params.getSamplingStartTimeIndex()
                                      : 0; // will be a 3D dataset

  // Set cuboid dimensions (subtract two corners (add 1) and use the appropriate component)
  DimensionSizes cuboidSize((mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).nx,
                            (mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).ny,
                            (mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx)).nz,
                            nSampledTimeSteps);

  // Set chunk size
  // If the size of the cuboid is bigger than 32 MB per time step, set the chunk to approx 8MB
  size_t nSlabs = 1; //at least one slab
  DimensionSizes cuboidChunkSize(cuboidSize.nx,
                                 cuboidSize.ny,
                                 cuboidSize.nz,
                                 (mReduceOp == ReduceOperator::kNone) ? 1 : 0);

  if (cuboidChunkSize.nElements() > (kChunkSize8MB * 4))
  {
    while (nSlabs * cuboidSize.nx * cuboidSize.ny < kChunkSize8MB) nSlabs++;
    cuboidChunkSize.nz = nSlabs;
  }

  // Indexed from 1
  const std::string datasetName = std::to_string(cuboidIdx + 1);

  hid_t dataset = mFile.createDataset(mGroup,
                                      datasetName,
                                      cuboidSize,
                                      cuboidChunkSize,
                                      Hdf5File::MatrixDataType::kFloat,
                                      params.getCompressionLevel());

  // Write dataset parameters
  mFile.writeMatrixDomainType(mGroup, datasetName, Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mGroup, datasetName, Hdf5File::MatrixDataType::kFloat);

  return dataset;
}//end of createCuboidDataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush the buffer to the file (to multiple datasets if necessary).
 */
void CuboidOutputStream::flushBufferToFile()
{
  DimensionSizes position (0, 0, 0, 0);
  DimensionSizes blockSize(0, 0, 0, 0);

  if (mReduceOp == ReduceOperator::kNone)
  {
    position.nt = mSampledTimeStep;
  }

  for (size_t cuboidIdx = 0; cuboidIdx < mCuboidsInfo.size(); cuboidIdx++)
  {
    blockSize    = mSensorMask.getBottomRightCorner(cuboidIdx) - mSensorMask.getTopLeftCorner(cuboidIdx);
    blockSize.nt = 1;

    mFile.writeHyperSlab(mCuboidsInfo[cuboidIdx].cuboidIdx,
                         position,
                         blockSize,
                         mHostBuffer + mCuboidsInfo[cuboidIdx].startingPossitionInBuffer);
  }

  mSampledTimeStep++;
}// end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

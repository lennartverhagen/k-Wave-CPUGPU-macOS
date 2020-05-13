/**
 * @file      IndexOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving data based on index senor mask into
 *            the output HDF5 file.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      29 August    2014, 10:10 (created) \n
 *            11 February  2020, 16:21 (revised)
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

#include <OutputStreams/IndexOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset, SourceMatrix, SensorMask and reduction operator together.
 */
IndexOutputStream::IndexOutputStream(Hdf5File&            file,
                                     const MatrixName&    datasetName,
                                     const RealMatrix&    sourceMatrix,
                                     const IndexMatrix&   sensorMask,
                                     const ReduceOperator reduceOp)
  : BaseOutputStream(file, datasetName, sourceMatrix, reduceOp),
    mSensorMask(sensorMask),
    mDataset(H5I_BADID),
    mSampledTimeStep(0),
    mEventSamplingFinished()
{
  // Create event for sampling
  cudaCheckErrors(cudaEventCreate(&mEventSamplingFinished));
}// end of IndexOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
IndexOutputStream::~IndexOutputStream()
{
  // Destroy sampling event
  cudaCheckErrors(cudaEventDestroy(mEventSamplingFinished));

  close();
  // free memory
  freeMemory();
}// end of ~IndexOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream, create a dataset, and allocate data for it.
 */
void IndexOutputStream::create()
{
  const size_t nSampledElementsPerStep = mSensorMask.size();

  const Parameters& params = Parameters::getInstance();

  // Derive dataset dimension sizes
  DimensionSizes datasetSize(nSampledElementsPerStep,
                             (mReduceOp == ReduceOperator::kNone)
                                ? params.getNt() - params.getSamplingStartTimeIndex() : 1,
                             1);

  // Set HDF5 chunk size
  DimensionSizes chunkSize(nSampledElementsPerStep, 1, 1);
  // For data bigger than 32 MB
  if (nSampledElementsPerStep > (kChunkSize8MB * 4))
  {
    chunkSize.nx = kChunkSize8MB; // Set chunk size to 8 MB
  }

  // Create a dataset under the root group
  mDataset = mFile.createDataset(mFile.getRootGroup(),
                                 mRootObjectName,
                                 datasetSize,
                                 chunkSize,
                                 Hdf5File::MatrixDataType::kFloat,
                                 params.getCompressionLevel());

  // Write dataset parameters
  mFile.writeMatrixDomainType(mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDataType::kFloat);

  // Sampled time step
  mSampledTimeStep = 0;

  // Set buffer size
  mBufferSize = nSampledElementsPerStep;

  // Allocate memory
  allocateMemory();
}// end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart and reload data.
 */
void IndexOutputStream::reopen()
{
  // Get parameters
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  mBufferSize = mSensorMask.size();

  // Allocate memory
   allocateMemory();

  // Reopen the dataset
  mDataset = mFile.openDataset(mFile.getRootGroup(), mRootObjectName);

  if (mReduceOp == ReduceOperator::kNone)
  { // Raw time series - just seek to the right place in the dataset
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex())
                          ? 0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }
  else
  { // Aggregated quantities - reload data
    mSampledTimeStep = 0;

    // Read only if it is necessary (it is anything to read).
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      // Since there is only a single time step in the dataset, I can read the whole dataset
      mFile.readCompleteDataset(mFile.getRootGroup(),
                                mRootObjectName,
                                DimensionSizes(mBufferSize, 1, 1),
                                mHostBuffer);

      // Send data to device
      copyToDevice();
    }
  }
}// end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample grid points, line them up in the buffer and apply reduction operator.
 */
void IndexOutputStream::sample()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kNone>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());

      // Record an event when the data has been copied over.
      cudaCheckErrors(cudaEventRecord(mEventSamplingFinished));

      break;
    }

    case ReduceOperator::kRms:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kRms>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());

      break;
    }

    case ReduceOperator::kMax:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kMax>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());
      break;
    }

    case ReduceOperator::kMin:
    {
      OutputStreamsCudaKernels::sampleIndex<ReduceOperator::kMin>
                                           (mDeviceBuffer,
                                            mSourceMatrix.getDeviceData(),
                                            mSensorMask.getDeviceData(),
                                            mSensorMask.size());
      break;
    }
  }// switch
}// end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush data for the times step. Only applicable on raw data series.
 */
void IndexOutputStream::flushRaw()
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

/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void IndexOutputStream::postProcess()
{
  // Run inherited method
  BaseOutputStream::postProcess();

  // When no reduction operator is applied, the data is flushed after every time step
  // which means it has been done before
  if (mReduceOp != ReduceOperator::kNone)
  {
    // Copy data from GPU matrix
    copyFromDevice();
    // Flush to disk
    flushBufferToFile();
  }
}// end of postProcess
//----------------------------------------------------------------------------------------------------------------------

/**
 * Checkpoint the stream.
 */
void IndexOutputStream::checkpoint()
{
  // Raw data has already been flushed, others has to be flushed here
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
 * Close stream (apply post-processing if necessary, flush data and close).
 */
void IndexOutputStream::close()
{
  // The dataset is still opened
  if (mDataset != H5I_BADID)
  {
    mFile.closeDataset(mDataset);
  }

  mDataset = H5I_BADID;
}// end of close
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Flush the buffer to the file.
 */
void IndexOutputStream::flushBufferToFile()
{
  mFile.writeHyperSlab(mDataset,
                       DimensionSizes(0, mSampledTimeStep, 0),
                       DimensionSizes(mBufferSize, 1, 1),
                       mHostBuffer);
  mSampledTimeStep++;
}// end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

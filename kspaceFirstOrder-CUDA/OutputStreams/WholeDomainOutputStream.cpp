/**
 * @file      WholeDomainOutputStream.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the class saving RealMatrix data into the output
 *            HDF5 file, e.g. p_max_all.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      28 August    2014, 11:15 (created) \n
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

#include <OutputStreams/WholeDomainOutputStream.h>
#include <Parameters/Parameters.h>

#include <OutputStreams/OutputStreamsCudaKernels.cuh>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Constructor - links the HDF5 dataset and SourceMatrix.
 */
WholeDomainOutputStream::WholeDomainOutputStream(Hdf5File&            file,
                                                 const MatrixName&    datasetName,
                                                 const RealMatrix&    sourceMatrix,
                                                 const ReduceOperator reduceOp)
  : BaseOutputStream(file, datasetName, sourceMatrix, reduceOp),
    mDataset(H5I_BADID),
    mSampledTimeStep(0)
{

}// end of WholeDomainOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destructor.
 */
WholeDomainOutputStream::~WholeDomainOutputStream()
{
  close();
  // Free memory
  freeMemory();
}// end of ~WholeDomainOutputStream
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create a HDF5 stream for the whole domain and allocate data for it.
 */
void WholeDomainOutputStream::create()
{
  DimensionSizes chunkSize(mSourceMatrix.getDimensionSizes().nx,
                           mSourceMatrix.getDimensionSizes().ny,
                           1);

  // Create a dataset under the root group
  mDataset = mFile.createDataset(mFile.getRootGroup(),
                                 mRootObjectName,
                                 mSourceMatrix.getDimensionSizes(),
                                 chunkSize,
                                 Hdf5File::MatrixDataType::kFloat,
                                 Parameters::getInstance().getCompressionLevel());

  // Write dataset parameters
  mFile.writeMatrixDomainType(mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDomainType::kReal);
  mFile.writeMatrixDataType  (mFile.getRootGroup(), mRootObjectName, Hdf5File::MatrixDataType::kFloat);

  // Set buffer size
  mBufferSize = mSourceMatrix.size();

  // Allocate memory
  allocateMemory();
}//end of create
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reopen the output stream after restart and reload data.
 */
void WholeDomainOutputStream::reopen()
{
  const Parameters& params = Parameters::getInstance();

  // Set buffer size
  mBufferSize = mSourceMatrix.size();

  // Allocate memory
  allocateMemory();

  // Open the dataset under the root group
  mDataset = mFile.openDataset(mFile.getRootGroup(), mRootObjectName);

  mSampledTimeStep = 0;
  if (mReduceOp == ReduceOperator::kNone)
  { // Seek in the dataset
    mSampledTimeStep = (params.getTimeIndex() < params.getSamplingStartTimeIndex())
                          ? 0 : (params.getTimeIndex() - params.getSamplingStartTimeIndex());
  }
  else
  { // Reload data
    // Read data from disk only if there were anything stored there (t_index > start_index)
    //(one step ahead)
    if (params.getTimeIndex() > params.getSamplingStartTimeIndex())
    {
      mFile.readCompleteDataset(mFile.getRootGroup(),
                                mRootObjectName,
                                mSourceMatrix.getDimensionSizes(),
                                mHostBuffer);
      // Send data to device
      copyToDevice();
    }
  }
}// end of reopen
//----------------------------------------------------------------------------------------------------------------------

/**
 * Sample all grid points, line them up in the buffer an flush to the disk unless a reduction operator is applied.
 */
void WholeDomainOutputStream::sample()
{
  switch (mReduceOp)
  {
    case ReduceOperator::kNone:
    {
      // Copy all data from decice to host (no need to use a kernel)
      // This violates the const prerequisite, however, this routine is still NOT used in the code
      const_cast<RealMatrix&>(mSourceMatrix).copyFromDevice();

      // We use here direct HDF5 offload using MEMSPACE - seems to be faster for bigger datasets
      const DimensionSizes datasetPosition(0, 0, 0, mSampledTimeStep); //4D position in the dataset

      DimensionSizes cuboidSize(mSourceMatrix.getDimensionSizes());// Size of the cuboid
      cuboidSize.nt = 1;

      mFile.writeCuboidToHyperSlab(mDataset,
                                   datasetPosition,
                                   DimensionSizes(0, 0, 0, 0), // position in the SourceMatrix
                                   cuboidSize,
                                   mSourceMatrix.getDimensionSizes(),
                                   mSourceMatrix.getHostData());

      // Move forward in time
      mSampledTimeStep++;

      break;
    }// case kNone

    case ReduceOperator::kRms:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kRms>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    }// case kRms

    case ReduceOperator::kMax:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kMax>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    }//case roMAX

    case ReduceOperator::kMin:
    {
      OutputStreamsCudaKernels::sampleAll<ReduceOperator::kMin>
                                         (mDeviceBuffer,
                                          mSourceMatrix.getDeviceData(),
                                          mSourceMatrix.size());
      break;
    }//case kMin
  }// switch
}// end of sample
//----------------------------------------------------------------------------------------------------------------------

/**
 * Apply post-processing on the buffer and flush it to the file.
 */
void WholeDomainOutputStream::postProcess()
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
void WholeDomainOutputStream::checkpoint()
{
  // Copy data on the CPU
  copyFromDevice();

  // Raw data has already been flushed, others has to be flushed here
  if (mReduceOp != ReduceOperator::kNone)
  {
    flushBufferToFile();
  }
}// end of checkpoint
//----------------------------------------------------------------------------------------------------------------------

/**
 * Close stream.
 */
void WholeDomainOutputStream::close()
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
void WholeDomainOutputStream::flushBufferToFile()
{
  DimensionSizes size = mSourceMatrix.getDimensionSizes();
  DimensionSizes position(0, 0, 0);

  // Not used for kNone now!
  if (mReduceOp == ReduceOperator::kNone)
  {
    position.nt = mSampledTimeStep;
    size.nt = mSampledTimeStep;
  }

  mFile.writeHyperSlab(mDataset, position, size, mHostBuffer);
  mSampledTimeStep++;
}// end of flushBufferToFile
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

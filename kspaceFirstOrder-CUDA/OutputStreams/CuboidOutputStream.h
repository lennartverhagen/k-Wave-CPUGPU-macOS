/**
 * @file      CuboidOutputStream.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file of classes responsible for storing output quantities based on the
 *            cuboid sensor mask into the output HDF5 file.
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

#ifndef CUBOID_OUTPUT_STREAM_H
#define CUBOID_OUTPUT_STREAM_H

#include <vector>
#include <cuda_runtime.h>

#include <OutputStreams/BaseOutputStream.h>

/**
 * @class   CuboidOutputStream
 * @brief   Output stream for quantities sampled by a cuboid corner sensor mask.
 * @details Output stream for quantities sampled by a cuboid corner sensor mask. This class writes data into separated
 *          datasets (one per cuboid) under a given dataset in the HDF5 file (time-series as well as aggregations).
 */
class CuboidOutputStream : public BaseOutputStream
{
  public:
    /// Default constructor not allowed.
    CuboidOutputStream() = delete;
    /**
     * @brief Constructor.
     *
     * @details The constructor links the HDF5 dataset, source (sampled matrix), sensor mask and the reduction operator
     *          together. The constructor DOES NOT allocate memory because the size of the sensor mask is not known at
     *          the time the instance of the class is being created.
     *
     * @param [in] file         - HDF5 file to write the output to.
     * @param [in] groupName    - HDF5 group name. This group contains datasets for particular cuboids.
     * @param [in] sourceMatrix - Source real matrix to be sampled.
     * @param [in] sensorMask   - Sensor mask with the cuboid coordinates.
     * @param [in] reduceOp     - Reduction operator.
     */
    CuboidOutputStream(Hdf5File&            file,
                       const MatrixName&    groupName,
                       const RealMatrix&    sourceMatrix,
                       const IndexMatrix&   sensorMask,
                       const ReduceOperator reduceOp);

    /// Copy constructor not allowed.
    CuboidOutputStream(const CuboidOutputStream&) = delete;

    /// Destructor.
    virtual ~CuboidOutputStream() override;

    /// Operator = not allowed.
    CuboidOutputStream& operator=(const CuboidOutputStream&) = delete;

    /// Create a HDF5 stream, allocate data for it, and open necessary datasets.
    virtual void create()      override;

    /// Reopen the output stream after restart and reload data.
    virtual void reopen()      override;

    /**
     * @brief   Sample grid points, line them up in the buffer, if necessary a reduce operator is applied.
     * @warning Data is not flushed, there is no sync.
     */
    virtual void sample()      override;

    /// Flush data to disk (for raw streams only).
    virtual void flushRaw()    override;

    /// Apply post-processing on the buffer and flush it to the file.
    virtual void postProcess() override;

    /// Checkpoint the stream.
    virtual void checkpoint()  override;

    /// Close stream.
    virtual void close()       override;

  protected:
    /**
     * @struct CuboidInfo
     * @brief  This structure holds information about a HDF5 dataset (one cuboid).
     */
    struct CuboidInfo
    {
      /// Idx of the dataset storing the given cuboid.
      hid_t  cuboidIdx;
      /// Having a single buffer for all cuboids, where this one starts.
      size_t startingPossitionInBuffer;
    };

    /**
     * @brief  Create a new dataset for a given cuboid specified by the index (order).
     * @param  [in] cuboidIdx - Index of the cuboid in the sensor mask.
     * @return Handle to the HDF5 dataset.
     */
    virtual hid_t createCuboidDataset(const size_t cuboidIdx);

    /// Flush the buffer to the file.
    virtual void flushBufferToFile();

    /// Sensor mask to sample data.
    const IndexMatrix&      mSensorMask;

    /// Vector keeping handles and positions of all cuboids.
    std::vector<CuboidInfo> mCuboidsInfo;

    /// Handle to a HDF5 dataset.
    hid_t  mGroup;

    /// Time step to store (N/A for aggregated).
    size_t mSampledTimeStep;

    /// Has the sampling finished?
    cudaEvent_t mEventSamplingFinished;
};// end of CuboidOutputStream
//----------------------------------------------------------------------------------------------------------------------

#endif /* CUBOID_OUTPUT_STREAM_H */


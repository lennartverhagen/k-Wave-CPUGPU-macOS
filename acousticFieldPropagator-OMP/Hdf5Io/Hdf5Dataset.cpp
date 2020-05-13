/**
 * @file Hdf5Io/Hdf5Dataset.cpp
 *
 * @brief Maintaining HDF5 datasets
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

#include <Hdf5Io/Hdf5Dataset.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


/**
 * @brief Constructor opening a dataset
 */
Hdf5Dataset::Hdf5Dataset(const Hdf5File& file,
                         const char*     name)
  : mDatasetName(name)
{
  mDatasetDesc = H5Dopen(file.mFileDesc, name, H5P_DEFAULT);
  if (mDatasetDesc < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to open dataset " + mDatasetName));
  }
  mDataspaceDesc = H5Dget_space(mDatasetDesc);
  if (mDataspaceDesc < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to obtain dataspace of dataset " + mDatasetName));
  }
}// end of Hdf5Dataset::Hdf5Dataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to get a dimensionality of the associated dataset
 */
std::size_t Hdf5Dataset::rank()
{
  int rank = H5Sget_simple_extent_ndims(mDataspaceDesc);
  if (rank < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to obtain dataspace rank of dataset " + mDatasetName));
  }
  return static_cast<std::size_t>(rank);
}// end of Hdf5Dataset::getRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to get size of the dataset
 */
std::vector<std::size_t> Hdf5Dataset::size()
{
  static_assert(sizeof(std::size_t) == sizeof(hsize_t),
                "std::size_t and hsize_t have different sizes and this implementation is malformed");
  static_assert(std::numeric_limits<std::size_t>::is_signed == std::numeric_limits<hsize_t>::is_signed,
                "std::size_t and hsize_t have different signnesses and this implementation is malformed");

  std::vector<std::size_t> dimensions;
  dimensions.resize(rank());

  herr_t status = H5Sget_simple_extent_dims(mDataspaceDesc, reinterpret_cast<hsize_t*>(dimensions.data()), nullptr);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to obtain dataspace dimensions of dataset " + mDatasetName));
  }
  return dimensions;
}// end of Hdf5Dataset::getRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method returning number of elements in the dataset
 */
std::size_t Hdf5Dataset::elementCount()
{
  std::size_t count = 1;
  for (auto& i : size())
  {
    count *= i;
  }
  return count;
}// end of Hdf5Dataset::getElementCount
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to read from the dataset
 */
void Hdf5Dataset::read(hid_t type,
                       hid_t memorySpace,
                       void* buffer)
{
  herr_t status = H5Dread(mDatasetDesc, type, memorySpace, mDataspaceDesc, H5P_DEFAULT, buffer);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to read from the dataset" + mDatasetName));
  }
}// end of Hdf5Dataset::read
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to write into the dataset
 */
void Hdf5Dataset::write(hid_t type, hid_t memorySpace, const void* buffer)
{
  herr_t status = H5Dwrite(mDatasetDesc, type, memorySpace, mDataspaceDesc, H5P_DEFAULT, buffer);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to write to the dataset" + mDatasetName));
  }
}// end of Hdf5Dataset::write
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to close the dataset
 */
void Hdf5Dataset::close()
{
  // the move assignment will take care of closing the original handles
  mDataspaceDesc = H5I_BADID;
  mDatasetDesc   = H5I_BADID;
}// end of Hdf5Dataset::close
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor
 */
Hdf5Dataset::Hdf5Dataset(const Hdf5File&      file,
                         const char*          name,
                         hid_t                type,
                         std::size_t          rank,
                         const hsize_t* const size)
    : mDatasetName(name)
{
  mDataspaceDesc                  = H5Screate_simple(rank, size, nullptr);
  Hdf5Id<H5Pclose> propertyListId = H5Pcreate(H5P_DATASET_CREATE);

  mDatasetDesc = H5Dcreate(file.mFileDesc, name, type, mDataspaceDesc, H5P_DEFAULT, propertyListId, H5P_DEFAULT);
  if (mDatasetDesc < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to create dataset " + mDatasetName));
  }
}// end of Hdf5Dataset::Hdf5Dataset
//----------------------------------------------------------------------------------------------------------------------

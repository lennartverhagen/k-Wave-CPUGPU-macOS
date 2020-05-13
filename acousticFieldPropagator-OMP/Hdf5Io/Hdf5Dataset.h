/**
 * @file Hdf5Io/Hdf5Dataset.h
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

#ifndef HDF5_DATASET_H
#define HDF5_DATASET_H

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Hdf5Io/Hdf5Error.h>
#include <Hdf5Io/Hdf5File.h>
#include <Hdf5Io/Hdf5Id.h>

/**
 * @brief RAII wrapper for HDF5 dataset
 *
 * Class that takes care of opening or creating datasets in the `Hdf5File`. It also keeps dataspace descriptor for the
 * entire dataset.
 */
class Hdf5Dataset
{
    /// Attribute needs to access the dataset descriptor on creation
    friend class Hdf5StringAttribute;

  public:
    /**
     * @brief Constructor opening a dataset
     *
     * Opens up a dataset and an associated dataspace with the provided name from the given file.
     *
     * @param[in] file – `Hdf5File` to open dataset from
     * @param[in] name – Name of the opened dataset
     * @throws std::runtime_error if the opening fails
     */
    Hdf5Dataset(const Hdf5File& file,
                const char*     name);

    /**
     * @brief Constructor creating a dataset
     *
     * Creates a new dataset with a given name in the given file. Type of the content needs to be specified along with
     * the size of the dataset. For the size, a container containing sizes in all the dimensions needs to be provided.
     * Size of the container itself will be a rank (dimensionality) of the created dataset.
     *
     * Dataspace is created for this dataspace, containing all elements.
     *
     * @tparam    Type – Type of the container to pass the `size`, automatically deduced
     * @param[in] file – `Hdf5File` to create dataset in
     * @param[in] name – Name of the created dataset
     * @param[in] type – Element type in the dataset
     * @param[in] size – Size of the dataset
     * @throws std::runtime_error if the creation fails
     */
    template<typename Type>
    Hdf5Dataset(const Hdf5File& file,
                const char*     name,
                hid_t           type,
                const Type&     size);

    /// Copy constructor not allowed
    Hdf5Dataset(const Hdf5Dataset&)            = delete;
    /// Operator = not allowed
    Hdf5Dataset& operator=(const Hdf5Dataset&) = delete;
    /// Move constructor not allowed
    Hdf5Dataset(Hdf5Dataset&& orig)            = delete;

    /**
     * @brief Method to check if the dataset is open
     * @returns True, if the dataset is open, false otherwise
     */
    bool isOpen() { return mDatasetDesc >= 0; }

    /**
     * @brief Method to close the dataset
     */
    void close();

    /**
     * @brief Method to get a dimensionality of the associated dataset
     * @returns Rank (number of dimensions) of the dataset
     */
    std::size_t rank();

    /**
     * @brief Method to get size of the dataset
     * @returns Vector containing dataset size in each direction
     */
    std::vector<std::size_t> size();

    /**
     * @brief Method returning number of elements in the dataset
     * @returns Number of elements in the dataset
     */
    std::size_t elementCount();

    /**
     * @brief Method to read from the dataset
     * @param[in]  type        – Target type of the data
     * @param[in]  memorySpace – Descriptor of the memory layout to write to
     * @param[out] buffer      – Memory the content will be written to
     * @throws std::runtime_error if reading fails
     */
    void read(hid_t type,
              hid_t memorySpace,
              void* buffer);

    /**
     * @brief Method to write into the dataset
     * @param[in] type        – Source type of the data
     * @param[in] memorySpace – Descriptor of the memory layout to read from
     * @param[in] buffer      – Memory the content will be read from
     * @throws std::runtime_error if writing fails
     */
    void write(hid_t       type,
               hid_t       memorySpace,
               const void* buffer);

    /**
     * @brief Method to restrict underlying dataspace descriptor
     * @tparam    Type1 – Type of the container to pass the `start`, automatically deduced
     * @tparam    Type2 – Type of the container to pass the `size`, automatically deduced
     * @param[in] start – Starting coordinates of the selection to make
     * @param[in] size  – Size of the selection to make
     */
    template<typename Type1,
             typename Type2>
    void select(const Type1& start,
                const Type2& size);

  private:
    /// Constructor to create the dataset, used internally
    Hdf5Dataset(const Hdf5File&      file,
                const char*          name,
                hid_t                type,
                std::size_t          rank,
                const hsize_t* const size);

    /// Dataset name (used for error reporting)
    std::string      mDatasetName;
    /// Dataset descriptor (id)
    Hdf5Id<H5Dclose> mDatasetDesc;
    /// Dataspace descriptor (id)
    Hdf5Id<H5Sclose> mDataspaceDesc;
};// end of Hdf5Dataset
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Template methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor creating a dataset
 */
template<typename Type>
inline Hdf5Dataset::Hdf5Dataset(const Hdf5File& file,
                                const char*     name,
                                hid_t           type,
                                const Type&     size)
    : Hdf5Dataset(file, name, type, size.size(), reinterpret_cast<const hsize_t*>(size.data()))
{
  static_assert(sizeof(typename Type::value_type) == sizeof(hsize_t),
                "hsize_t has a different size and this implementation is malformed");
  static_assert(std::numeric_limits<typename Type::value_type>::is_signed == std::numeric_limits<hsize_t>::is_signed,
                "hsize_t has a different signness and this implementation is malformed");
}// end of Hdf5Dataset::Hdf5Dataset
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to restrict underlying dataspace descriptor
 */
template<typename Type1,
         typename Type2>
void Hdf5Dataset::select(const Type1& start,
                         const Type2& size)
{
  static_assert(sizeof(hsize_t) == sizeof(typename Type1::value_type),
                "The provided container is of an incompatible type");
  static_assert(std::numeric_limits<hsize_t>::is_signed == std::numeric_limits<typename Type1::value_type>::is_signed,
                "The provided container is of an incompatible type");
  static_assert(sizeof(hsize_t) == sizeof(typename Type2::value_type),
                "The provided container is of an incompatible type");
  static_assert(std::numeric_limits<hsize_t>::is_signed == std::numeric_limits<typename Type2::value_type>::is_signed,
                "The provided container is of an incompatible type");

  if (start.size() != rank() || size.size() != rank())
  {
    throw std::runtime_error("Selection parameters do not match the rank of the dataset " + mDatasetName);
  }

  herr_t status = H5Sselect_hyperslab(mDataspaceDesc, H5S_SELECT_SET, reinterpret_cast<const hsize_t*>(start.data()),
                                      nullptr, reinterpret_cast<const hsize_t*>(size.data()), nullptr);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to restrict dataspace selection"));
  }
}// end of Hdf5Dataset::select
//----------------------------------------------------------------------------------------------------------------------

#endif /* HDF5_DATASET_H */

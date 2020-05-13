/**
 * @file Hdf5Io/Hdf5MemSpace.h
 *
 * @brief Constructing memory spaces in HDF5
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
 * Created: 2017-09-28 15:50\n
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

#ifndef HDF5_MEM_SPACE
#define HDF5_MEM_SPACE

#include <limits>
#include <stdexcept>
#include <vector>

#include <hdf5.h>

#include <Hdf5Io/Hdf5Error.h>
#include <Hdf5Io/Hdf5Id.h>

/**
 * @brief RAII wrapper for HDF5 memory space
 *
 * Class that takes care of opening or creating memory space descriptors.
 */
class Hdf5MemSpace
{
  public:
    /**
     * @brief Constructor creating HDF5 memory space
     *
     * @tparam    Type – Type of the container to pass the `size`, automatically deduced
     * @param[in] size – Size of the memory space
     * @throws std::runtime_error if the memspace creation or restriction fails
     */
    template<typename Type>
    Hdf5MemSpace(const Type& size)
    {
      static_assert(sizeof(hsize_t) == sizeof(typename Type::value_type),
                    "The provided container is of an incompatible type");
      static_assert(std::numeric_limits<hsize_t>::is_signed ==
                        std::numeric_limits<typename Type::value_type>::is_signed,
                    "The provided container is of an incompatible type");

      mSpaceDesc = H5Screate_simple(size.size(), reinterpret_cast<const hsize_t*>(size.data()), nullptr);
      if (mSpaceDesc < 0)
      {
        throw std::runtime_error(getHdf5ErrorString("Failed to create memory space descriptor"));
      }
    }

    /**
     * @brief Constructor creating HDF5 memory space with a selection
     *
     * @tparam    Type1     – Type of the container to pass the `size`, automatically deduced
     * @tparam    Type2     – Type of the container to pass the `selection`, automatically deduced
     * @param[in] size      – Size of the memory space
     * @param[in] selection – Size of the memory space to restrict the selection onto
     * @throws std::runtime_error if the memspace creation or restriction fails
     */
    template<typename Type1, typename Type2>
    Hdf5MemSpace(const Type1& size,
                 const Type2& selection)
      : Hdf5MemSpace(size)
    {
      static_assert(sizeof(hsize_t) == sizeof(typename Type2::value_type),
                    "The provided container is of an incompatible type");
      static_assert(std::numeric_limits<hsize_t>::is_signed ==
                        std::numeric_limits<typename Type2::value_type>::is_signed,
                    "The provided container is of an incompatible type");

      if (size.size() != selection.size())
      {
        throw std::runtime_error("The size of the memory space and its selection have different ranks");
      }

      const std::vector<hsize_t> start(size.size(), 0);
      herr_t status = H5Sselect_hyperslab(mSpaceDesc, H5S_SELECT_SET, start.data(), nullptr,
                                          reinterpret_cast<const hsize_t*>(selection.data()), nullptr);
      if (status < 0)
      {
        throw std::runtime_error(getHdf5ErrorString("Failed to restrict memory space selection"));
      }
    }

    /**
     * @brief Method returning underlying space descriptor (ID)
     * @returns Space descriptor
     */
    hid_t space() { return mSpaceDesc; }

  private:
    /// Memspace descriptor
    Hdf5Id<H5Sclose> mSpaceDesc;
};// end of Hdf5MemSpace
//----------------------------------------------------------------------------------------------------------------------

#endif /* HDF5_MEM_SPACE */

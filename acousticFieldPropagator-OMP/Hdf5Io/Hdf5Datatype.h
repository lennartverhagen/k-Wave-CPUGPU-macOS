/**
 * @file Hdf5Io/Hdf5Datatype.h
 *
 * @brief Implementation of a class providing run-time mapping to HDF5 native datatypes
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
 * Created: 2020-02-07 15:30\n
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

#ifndef HDF5DATATYPE_H
#define HDF5DATATYPE_H

#include <cstddef>
#include <hdf5.h>

/**
 * @brief Class providing compile-time HDF5 type assignment
 *
 * Specializations of this class are supposed to define a static constant `kType`. This allows the selection of the
 * correct HDF5 type identifier to be selected based on a typename resolved at compilation time.
 *
 * For example, to obtain a correct HD5T constant for a type `Type`, use `Hdf5Datatype<typename Type>::kType`.
 */
template<typename Type>
struct Hdf5Datatype
{};//end of Hdf5Datatype
//----------------------------------------------------------------------------------------------------------------------

/// Specialization of `Hdf5Datatype` for `std::size_t` type.
template<>
struct Hdf5Datatype<std::size_t>
{
  /// kType in this specialization initialized to H5T_NATIVE_ULLONG at runtime
  static const hid_t kType;
};// end of Hdf5Datatype<std::size_t>
//----------------------------------------------------------------------------------------------------------------------

/// Specialization of `Hdf5Datatype` for `float` type.
template<>
struct Hdf5Datatype<float>
{
  /// kType in this specialization initialized to H5T_NATIVE_FLOAT at runtime
  static const hid_t kType;
};// end of Hdf5Datatype<float>
//----------------------------------------------------------------------------------------------------------------------

#endif // HDF5DATATYPE_H

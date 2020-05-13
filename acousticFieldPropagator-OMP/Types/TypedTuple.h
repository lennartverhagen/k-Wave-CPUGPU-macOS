/**
 * @file Types/TypedTuple.h
 *
 * @brief Wrapper for std::array with named accessors
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

#ifndef TYPED_TUPLE_H
#define TYPED_TUPLE_H

#include <array>

/**
 * @brief Tuple object with members of the same type
 *
 * This class provides a wrapper on top of std::array with named access methods x(), y(), z() t(). It was designed to
 * be used for storing sizes and positions of multidimensional arrays.
 *
 * @tparam Type  – Underlying type
 * @tparam kSize – Size of the tuple
 */
template<typename    Type,
         std::size_t kSize>
class TypedTuple : public std::array<Type, kSize>
{
  public:
    /**
     * @brief Constructor
     *
     * Templated constructor to be able to construct the object in the most intuitive ways.
     *
     * @param[in] args – Parameters to construct the tuple with.
     */
    template<typename... Args>
    TypedTuple(Args... args) : std::array<Type, kSize>(args...)
    {}

    /**
     * @brief Access element _x_
     * @returns First element of the tuple
     */
    typename std::array<Type, kSize>::reference x()
    {
      static_assert(kSize > 0, "You cannot access x() within a tuple of less than 1 element");
      return (*this)[0];
    }

    /**
     * @brief Access element _x_, immutable version
     * @returns First element of the tuple
     */
    typename std::array<Type, kSize>::const_reference x() const
    {
      static_assert(kSize > 0, "You cannot access x() within a tuple of less than 1 element");
      return (*this)[0];
    }

    /**
     * @brief Access element _y_
     * @returns Second element of the tuple
     */
    typename std::array<Type, kSize>::reference y()
    {
      static_assert(kSize > 1, "You cannot access y() within a tuple of less than 2 elements");
      return (*this)[1];
    }

    /**
     * @brief Access element _y_, immutable version
     * @returns Second element of the tuple
     */
    typename std::array<Type, kSize>::const_reference y() const
    {
      static_assert(kSize > 1, "You cannot access y() within a tuple of less than 2 elements");
      return (*this)[1];
    }

    /**
     * @brief Access element _z_
     * @returns Third element of the tuple
     */
    typename std::array<Type, kSize>::reference z()
    {
      static_assert(kSize > 2, "You cannot access z() within a tuple of less than 3 elements");
      return (*this)[2];
    }

    /**
     * @brief Access element _z_, immutable version
     * @returns Third element of the tuple
     */
    typename std::array<Type, kSize>::const_reference z() const
    {
      static_assert(kSize > 2, "You cannot access z() within a tuple of less than 3 elements");
      return (*this)[2];
    }

    /**
     * @brief Access element _t_
     * @returns Last element of the tuple
     */
    typename std::array<Type, kSize>::reference t()
    {
      static_assert(kSize > 0, "You cannot access t() within a tuple of less than 1 element");
      return *this->rbegin();
    }

    /**
     * @brief Access element _t_, immutable version
     * @returns Last element of the tuple
     */
    typename std::array<Type, kSize>::const_reference t() const
    {
      static_assert(kSize > 0, "You cannot access t() within a tuple of less than 1 element");
      return *this->crbegin();
    }

    /**
     * @brief Return the product of elements
     *
     * When the TypedTuple contains dimension sizes, this method returns numbers of elements in the domain it describes.
     *
     * @returns Product of all elements in the tuple
     */
    Type product() const
    {
      Type result = Type(1);
      for (const auto& element : *this)
      {
        result *= element;
      }
      return result;
    }
};// end of TypedTuple
//----------------------------------------------------------------------------------------------------------------------

#endif /* TYPED_TUPLE_H */

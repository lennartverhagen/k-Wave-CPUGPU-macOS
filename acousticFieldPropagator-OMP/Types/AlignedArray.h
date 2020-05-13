/**
 * @file Types/AlignedArray.h
 *
 * @brief Aligned std::array-like container
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

#ifndef ALIGNED_ARRAY_H
#define ALIGNED_ARRAY_H

#include <cstddef>
#include <new>
#include <stdexcept>
#include <xmmintrin.h>

/**
 * @brief Dynamically allocated array-like container with aligned storage
 *
 * This class implements array-like container with std-like interface. The size of the array is constant and passed
 * as a constructor parameter. Objects are properly constructed with placement new, thus supporting non-trivial
 * data types. Arguments can be provided to the constructor of the underlying type, a default constructor is used
 * instead.
 *
 * @tparam Type   – Data type of the stored objects
 * @tparam kAlign – Required storage alignment
 */
template<typename   Type,
        std::size_t kAlign>
class AlignedArray
{
    using iterator       = Type*;
    using const_iterator = const Type*;
    using size_type      = std::size_t;

  public:
    /**
     * @brief Constructor
     *
     * All the objects are constructed using a constructor without parameters. This is usually a default constructor,
     * with POD types no initialization is performed.
     *
     * @param[in] size – Size of the created array
     * @throws std::bad_alloc if the space allocation fails
     */
    AlignedArray(size_type size);

    /**
     * @brief Constructor with parameters
     *
     * Overloaded variant of the AlignedArray constructor. Additional parameters are passed to the constructor of the
     * underlying objects, allowing for copying, initialization or conversions.
     *
     * @param[in] size – Size of the created array
     * @param[in] args – Parameters to construct elements with
     * @throws std::bad_alloc if the space allocation fails
     */
    template<typename... Args>
    AlignedArray(size_type size, Args... args);

    /// Copy constructor not allowed
    AlignedArray(const AlignedArray&) = delete;
    /// Operator = not allowed
    AlignedArray& operator=(const AlignedArray&) = delete;

    /**
     * @brief Move constructor
     *
     * Allows for cheap moving of the AlignedArray object.
     *
     * @param[in,out] orig – Original object
     */
    AlignedArray(AlignedArray&& orig) : mData(orig.mData), mSize(orig.mSize)
    {
      orig.mData = nullptr;
      orig.mSize = 0;
    }

    /**
     * @brief Destructor
     *
     * Destructs all the elements in the array and deallocates the space.
     */
    ~AlignedArray();

    /// Random access iterator to the first element
    iterator begin()              { return mData; }
    /// Random access iterator to the first element, constant overload
    const_iterator begin()  const { return mData; }
    /// Random access iterator to the first element, constant variant
    const_iterator cbegin() const { return mData; }
    /// Random access iterator past the last element
    iterator end()                { return mData + mSize; }
    /// Random access iterator past the last element, constant overload
    const_iterator end()    const { return mData + mSize; }
    /// Random access iterator past the last element, constant variant
    const_iterator cend()   const { return mData + mSize; }

    /**
     * @brief Element accessor
     *
     * No bound checking is performed.
     *
     * @param[in] index – Index of the requested element
     * @returns Reference to the element at the given index
     */
    Type& operator[](size_type index) { return *(mData + index); }

    /**
     * @brief Element accessor, constant overload
     *
     * No bound checking is performed.
     *
     * @param[in] index – Index of the requested element
     * @returns Constant reference to the element at the given index
     */
    const Type& operator[](size_type index) const { return *(mData + index); }

    /**
     * @brief Method returning the size of the array
     * @returns Number of elements in the array
     */
    size_type size() const { return mSize; }

  private:
    /// Pointer to the allocated storage.
    Type*           mData;
    /// Size of the allocated storage, in number of elements.
    const size_type mSize;
};// end of AlignedArray
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Template methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor
 */
template<typename    Type,
         std::size_t kAlign>
inline AlignedArray<Type, kAlign>::AlignedArray(size_type size)
  : mData(nullptr),
    mSize(size)
{
  // space allocation
  mData = static_cast<Type*>(_mm_malloc(size * sizeof(Type), alignof(Type) > kAlign ? alignof(Type) : kAlign));
  if (!mData)
  {
    throw std::bad_alloc();
  }

  // NUMA first touch policy and object construction
  #pragma omp parallel for schedule(static)
  for (size_type i = 0; i < mSize; ++i)
  {
    ::new (mData + i) Type();
  }
}// end of AlignedArray::AlignedArray
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Constructor with parameters
 */
template<typename Type,
         std::size_t kAlign>
template<typename... Args>
inline AlignedArray<Type, kAlign>::AlignedArray(size_type size, Args... args)
  : mData(nullptr),
    mSize(size)
{
  // space allocation
  mData = static_cast<Type*>(_mm_malloc(size * sizeof(Type), alignof(Type) > kAlign ? alignof(Type) : kAlign));
  if (!mData)
  {
    throw std::bad_alloc();
  }

  // NUMA first touch policy and object construction
  #pragma omp parallel for schedule(static)
  for (size_type i = 0; i < mSize; ++i)
  {
    ::new (mData + i) Type(args...);
  }
}// end of AlignedArray::AlignedArray
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Destructor
 */
template<typename Type,
        std::size_t kAlign>
inline AlignedArray<Type, kAlign>::~AlignedArray()
{
  // object destruction
  for (auto& item : *this)
  {
    item.~Type();
  }

  // space deallocation
  _mm_free(static_cast<void*>(mData));
}// end of AlignedArray::~AlignedArray
//----------------------------------------------------------------------------------------------------------------------

#endif /* ALIGNED_ARRAY_H */

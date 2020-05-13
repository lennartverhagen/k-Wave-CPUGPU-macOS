/**
 * @file Types/ArrayBuffer.h
 *
 * @brief Statically-allocated std::basic_ostream buffer object
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
 * Created: 2017-03-22 18:23\n
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

#ifndef ARRAYBUFFER_H
#define ARRAYBUFFER_H

#include <array>
#include <streambuf>

/**
 * @brief Class meant to be used as a backend storage for std::basic_ostream
 *
 * With this class, it is possible to construct std::basic_ostream object and use formatting input
 * methods (operator<<) to a statically allocated storage.
 *
 * Underlying buffer has a constant size determined by the template argument. After the limit is
 * reached, an exception is thrown by the standard library by default.
 *
 * It is not possible to extract the stored information using the formatted output methods,
 * the resulting string can be obtained using the implemented random access methods.
 *
 * @tparam kSize    – Size of the underlying buffer, in number of CharT elements
 * @tparam CharType – Type of the character used in the buffer
 */
template<std::size_t kSize,
         typename    CharType = char>
class ArrayBuffer : public std::basic_streambuf<CharType>
{
    using char_type      = CharType;
    using const_iterator = const char_type*;
    using size_type      = std::size_t;

  public:
    /**
     * @brief ArrayBuffer constructor
     *
     * Initializes the object.
     */
    ArrayBuffer()
    {
      this->setp(mBuffer.data(), mBuffer.data() + kSize);
      this->setg(nullptr, nullptr, nullptr);
    }

    /**
     * @brief Copy constructor
     *
     * Buffers can be copied and this is the only way of moving them in the memory. As this involves
     * a deep copy of the content, copying is discouraged due to performance reasons.
     *
     * @param[in] orig – ArrayBuffer to copy the content from
     */
    ArrayBuffer(const ArrayBuffer& orig)
    {
      std::copy(orig.begin(), orig.end(), mBuffer.data());
      this->setp(mBuffer.data() + orig.size(), mBuffer.data() + kSize);
      this->setg(nullptr, nullptr, nullptr);
    }

    // semantically, it does not make sense to move this object as its storage is static
    ArrayBuffer(ArrayBuffer&&) = delete;

    /// Method providing random access iterator to the beginning of the resulting buffer
    const_iterator begin() const { return mBuffer.data(); }
    /// Method providing random access iterator to the end of the resulting buffer
    const_iterator end()   const { return this->pptr(); }

    /**
     * @brief Method returning the current length of the stored string
     * @returns Size of the stored string, in the number of character elements
     */
    size_type size()       const { return end() - begin(); }

  private:
    /// Array used as a storage for the string
    std::array<CharType, kSize> mBuffer;
};// end of ArrayBuffer
//----------------------------------------------------------------------------------------------------------------------

#endif /* ARRAYBUFFER_H */

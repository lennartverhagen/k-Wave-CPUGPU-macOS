/**
 * @file Utils/Terminal.h
 *
 * @brief Column-formatted text output
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

#ifndef TERMINAL_H
#define TERMINAL_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <iomanip>
#include <ostream>

#include <Types/ArrayBuffer.h>

/// Type alias for a function manipulating std::ios_base objects
using iomanip = std::ios_base& (*)(std::ios_base&);

/**
 * @brief Class providing an interface to a simple, column-oriented formatted output
 *
 * This class is a template class that provides a methods useful for printing output in a column.
 * It supports one and two-column formatting. Most of the measures and properties are template-
 * based and thus evaluated at the compile time.
 *
 * @tparam kWidth   – Width of the output column
 * @tparam kBufSize – Size of the underlying buffers used to format a single "line"
 */
template<std::size_t kWidth = 120,
         std::size_t kBufSize = 1024>
class Terminal
{
    using char_type      = std::ostream::char_type;
    using iterator       = char_type*;
    using const_iterator = const char_type*;

    /**
     * @brief Proxy object taking care of single-column formatted printing
     *
     * @tparam kManip – Manipulator that will be applied to the output
     */
    template<iomanip kManip>
    class SimpleProxy
    {
        /// friend class relationship to provide access to the constructor
        friend class Terminal;

      public:
        /// proxy object cannot be copied
        SimpleProxy(const SimpleProxy&) = delete;

        /**
         * @brief Move constructor
         *
         * It is possible to move the output proxy object, but it is discouraged as it involves
         * copying the underlying buffer. The method is mainly provided to allow move semantics,
         * hoping that the compiler applies return value optimization and no move is actually
         * performed (copy elision).
         *
         * @param[in] orig – Object to move from
         */
        SimpleProxy(SimpleProxy&& orig)
          : mParent(orig.mParent),
            mBuffer(orig.mBuffer),
            mStream(&mBuffer)
        {
          orig.mParent = nullptr;
        }

        /**
         * @brief Proxy object destruction
         *
         * Upon destruction, content accumulated in the underlying buffer is printed out to
         * a destination stream, taking the column width into an account.
         */
        ~SimpleProxy();

        /**
         * @brief Method providing formatted output
         *
         * It is possible to use all of the standard formatted output techniques. Be aware that
         * the output will be reformatted once more to fit into the column. This means that you
         * should avoid alignment modifiers and additional white-space characters, unless you are
         * sure what are you doing.
         *
         * @tparam    Type – Argument type
         * @param[in] arg  – Object to be printed
         * @returns Reference to self
         */
        template<typename Type>
        SimpleProxy& operator<<(Type arg)
        {
          mStream << arg;
          return *this;
        }

      private:
        /**
         * @brief SimpleProxy constructor
         * @param[in] parent – Associated Terminal object
         */
        SimpleProxy(Terminal& parent) : mParent(&parent), mStream(&mBuffer) {}
        /// Pointer to the associated terminal, nullptr if the object was moved
        Terminal* mParent;
        /// Underlying buffer for the output
        ArrayBuffer<kBufSize, char_type> mBuffer;
        /// Output stream object allowing the formatted output
        std::basic_ostream<char_type>    mStream;
    };// end of Terminal::SimpleProxy
    //-----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Proxy object taking care of double-column formatted printing
     *
     * @tparam kMiddlePoint – Index specifying the division point between the columns
     * @tparam kLeftManip   – Stream manipulator applied for the left column
     * @tparam kRightManip  – Stream manipulator applied for the right column
     */
    template<std::size_t kMiddlePoint, iomanip kLeftManip, iomanip kRightManip>
    class SplitProxy
    {
        // friend class relationship to provide access to the constructor
        friend class Terminal;

      public:
        // proxy object cannot be copied
        SplitProxy(const SplitProxy&) = delete;

        /**
         * @brief Move constructor
         *
         * It is possible to move the output proxy object, but it is discouraged as it involves
         * copying the underlying buffers. The method is mainly provided to allow move semantics,
         * hoping that the compiler applies return value optimization and no move is actually
         * performed (copy elision).
         *
         * @param[in] orig – Object to move from
         */
        SplitProxy(SplitProxy&& orig)
            : mParent(orig.mParent), mLeftBuffer(orig.mRightBuffer), mRightBuffer(orig.mRightBuffer),
              mLeftStream(&mLeftBuffer), mRightStream(&mRightBuffer)
        {
          orig.mParent = nullptr;
        }

        /**
         * @brief Proxy object destruction
         *
         * Upon destruction, content accumulated in the underlying buffers is printed out to
         * a destination stream, taking the column widths into an account.
         */
        ~SplitProxy();

        /**
         * @brief Method providing formatted output to the left column
         *
         * It is possible to use all of the standard formatted output techniques. Be aware that
         * the output will be reformatted once more to fit into the column. This means that you
         * should avoid alignment modifiers and additional white-space characters, unless you are
         * sure what are you doing.
         *
         * @tparam    Type – Argument type
         * @param[in] arg  – Object to be printed
         * @returns Reference to self
         */
        template<typename Type>
        SplitProxy& operator<<(Type arg)
        {
          mLeftStream << arg;
          return *this;
        }

        /**
         * @brief Method providing formatted output to the right column
         *
         * It is possible to use all of the standard formatted output techniques. Be aware that
         * the output will be reformatted once more to fit into the column. This means that you
         * should avoid alignment modifiers and additional white-space characters, unless you are
         * sure what are you doing.
         *
         * @tparam    Type – Argument type
         * @param[in] arg  – Object to be printed
         * @returns Reference to self
         */
        template<typename Type>
        SplitProxy& operator>>(Type arg)
        {
          mRightStream << arg;
          return *this;
        }

      private:
        /**
         * @brief SplitProxy constructor
         * @param[in] parent – Associated Terminal object
         */
        SplitProxy(Terminal& parent) : mParent(&parent), mLeftStream(&mLeftBuffer), mRightStream(&mRightBuffer)
        {
          static_assert(kMiddlePoint > 8, "First column of the split terminal is too narrow");
          static_assert(kWidth > kMiddlePoint, "The middle cannot be outside of terminal width");
          static_assert(kWidth - kMiddlePoint > 8 + 1, "Second column of the split terminal is too narrow");
        }
        /// Pointer to the associated terminal, nullptr if the object was moved
        Terminal* mParent;
        /// Underlying buffer for the left column output
        ArrayBuffer<kBufSize / 2, char_type> mLeftBuffer;
        /// Underlying buffer for the right column output
        ArrayBuffer<kBufSize / 2, char_type> mRightBuffer;
        /// Output stream object allowing the formatted output to the left column
        std::basic_ostream<char_type> mLeftStream;
        /// Output stream object allowing the formatted output to the right column
        std::basic_ostream<char_type> mRightStream;
    };// end of Terminal::SplitProxy
    //-----------------------------------------------------------------------------------------------------------------

  public:
    /**
     * @brief Terminal constructor
     *
     * Construct a Terminal object for the given output stream
     *
     * @param[in] target – Stream that all the content will be printed to
     */
    Terminal(std::ostream& target) : mStream(target)
    {
      static_assert(kWidth > 20, "Terminal must be wider than 20 characters");
    }
    /// Terminal destructor
    ~Terminal() {}

    /// Method printing a Terminal-wide separator made from hyphens
    void printSeparator();
    /// Method printing a Terminal-wide separator made from equality signs
    void printBoldSeparator();

    /**
     * @brief Method to obtain a single column output proxy object
     *
     * The column will be left-aligned by default.
     *
     * @tparam kManip – Stream manipulator to be applied to the column, std::left by default
     * @returns Single column proxy object
     */
    template<iomanip kManip = std::left>
    SimpleProxy<kManip> print1C()
    {
      return SimpleProxy<kManip>(*this);
    }

    /**
     * @brief Method to obtain a double column output proxy object
     *
     * Left column will be left-aligned, right column right-aligned by default.
     *
     * @tparam kMiddlePoint – Division point between the columns
     * @tparam kLeftManip   – Stream manipulator to be applied to the left column, std::left by default
     * @tparam kRightManip  – Stream manipulator to be applied to the right column, std::left by default
     */
    template<std::size_t kMiddlePoint,
             iomanip     kLeftManip = std::left,
             iomanip     kRightManip = std::right>
    SplitProxy<kMiddlePoint, kLeftManip, kRightManip> print2C()
    {
      return SplitProxy<kMiddlePoint, kLeftManip, kRightManip>(*this);
    }

    /**
     * @brief Method to obtain a double column output proxy object
     *
     * Left column will be left-aligned, right column right-aligned by default. Columns will be sized evenly.
     *
     * @tparam kLeftManip  – Stream manipulator to be applied to the left column, std::left by default
     * @tparam kRightManip – Stream manipulator to be applied to the right column, std::left by default
     */
    template<iomanip kLeftManip  = std::left,
             iomanip kRightManip = std::right>
    SplitProxy<kWidth / 2, kLeftManip, kRightManip> print2C11()
    {
      return SplitProxy<kWidth / 2, kLeftManip, kRightManip>(*this);
    }

    /**
     * @brief Method to obtain a double column output proxy object
     *
     * Left column will be left-aligned, right column right-aligned by default. Right column will be twice
     * as wide compared to the left column.
     *
     * @tparam kLeftManip  – Stream manipulator to be applied to the left column, std::left by default
     * @tparam kRightManip – Stream manipulator to be applied to the right column, std::left by default
     */
    template<iomanip kLeftManip  = std::left,
             iomanip kRightManip = std::right>
    SplitProxy<kWidth / 3, kLeftManip, kRightManip> print2C12()
    {
      return SplitProxy<kWidth / 3, kLeftManip, kRightManip>(*this);
    }

    /**
     * @brief Method to obtain a double column output proxy object
     *
     * Left column will be left-aligned, right column right-aligned by default. Left column will be twice
     * as wide compared to the right column.
     *
     * @tparam kLeftManip  – Stream manipulator to be applied to the left column, std::left by default
     * @tparam kRightManip – Stream manipulator to be applied to the right column, std::left by default
     */
    template<iomanip kLeftManip  = std::left,
             iomanip kRightManip = std::right>
    SplitProxy<kWidth * 2 / 3, kLeftManip, kRightManip> print2C21()
    {
      return SplitProxy<kWidth * 2 / 3, kLeftManip, kRightManip>(*this);
    }

  private:
    /**
     * @brief Method implementing pretty line printing
     *
     * Method copies the given string until either end of the string or specified width (template
     * parameter) is reached. This method is word-aware, thus preferring breaks when encountering
     * a white space. It also supports word hyphenation suggested by '|' character.
     *
     * For this reason, all the '|' characters are skipped and not printed, they are taken as a hint
     * on where to split the words. If the word is split, hyphen is inserted at the end.
     *
     * All the white-spaces that occurs at the end are skipped, ensuring that a new call to this
     * method will start with a printable character.
     *
     * @tparam        kMaxLength – Maximum number of characters to print out to the stream
     * @param[in,out] position   – Current position in the input string
     * @param[in]     end        – End of the input string
     */
    template<std::size_t kMaxLength>
    void prettyPrint(const_iterator& position,
                     const_iterator  end);
    /// Reference to the output stream
    std::ostream& mStream;
};// end of Terminal
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Template methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Method printing a Terminal-wide separator made from hyphens
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
void Terminal<kWidth, kBufSize>::printSeparator()
{
  mStream << std::setfill('-') << std::setw(kWidth) << "" << std::endl << std::setfill(' ');
}// end of Terminal::printSeparator
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method printing a Terminal-wide separator made from equality signs
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
void Terminal<kWidth, kBufSize>::printBoldSeparator()
{
  mStream << std::setfill('=') << std::setw(kWidth) << "" << std::endl << std::setfill(' ');
}// end of Terminal::printBoldSeparator
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method implementing pretty line printing
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
template<std::size_t kMaxLength>
void Terminal<kWidth, kBufSize>::prettyPrint(typename Terminal<kWidth, kBufSize>::const_iterator& position,
                                             typename Terminal<kWidth, kBufSize>::const_iterator  end)
{
  // store the beginning position for the wrap logic
  const_iterator beginning = position;
  // + 1 for the terminating null character
  std::array<char_type, kMaxLength + 1> buffer;
  auto bufferPosition = buffer.begin();

  // handle the edge situation with no input, this would mess up the algorithm
  if (position >= end)
  {
    mStream << "";
    return;
  }

  // try to find the first wrap point
  const_iterator wrap_point = position + 1;
  // wrap point is either the end of the input, white character or '|'
  while (wrap_point < end && !std::isspace(*wrap_point) && *wrap_point != '|')
  {
    ++wrap_point;
  }
  // if the wrap point is '|', we will need an extra character to store the hyphen
  // handle the end of the input!
  size_t extra = wrap_point < end && *wrap_point == '|';

  while (position < end && wrap_point - position + extra < buffer.end() - bufferPosition)
  {
    // as long as we can fit the input up to the wrap point into the buffer, let's do it
    // copy up to the found wrap point, leave out the '|'
    while (position < wrap_point)
    {
      if (*position == '|')
      {
        position++;
      }
      else
      {
        *(bufferPosition++) = *(position++);
      }
    }

    // now look for the following wrap point, the same way as above
    wrap_point = position + 1;
    while (wrap_point < end && !std::isspace(*wrap_point) && *wrap_point != '|')
    {
      ++wrap_point;
    }
    extra = wrap_point < end && *wrap_point == '|';
  }

  // we either ran out of the input or cannot fit more data into the buffer
  // check for the no wrap point available situation
  if (position == beginning)
  {
    // we do have input data, because position < end at the beginning
    // this also means the wrap point is at the end and input is longer than the buffer
    // so we force a wrap because there is anything else to do
    std::copy_n(position, kMaxLength, buffer.begin());
    bufferPosition = buffer.end() - 1;
    position += kMaxLength;
  }
  else if (position < end && *position == '|')
  {
    // we ended up on a separator mark and there is space reserved for a hyphen in the output buffer
    // so let's use it
    *(bufferPosition++) = '-';
  }

  // since we're breaking the line at this point, skip all the remaining white-spaces and marks
  while (position < end && (std::isspace(*position) || *position == '|'))
  {
    ++position;
  }

  // append the null character at the end of our buffer
  *bufferPosition = '\0';
  // and print it out
  mStream << buffer.data();
}// end of Terminal::prettyPrint
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Destructor
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
template<iomanip kManip>
Terminal<kWidth, kBufSize>::SimpleProxy<kManip>::~SimpleProxy()
{
  // correctly handle moved object, we don't want to duplicate the output
  if (!mParent)
  {
    return;
  }
  const_iterator current = mBuffer.begin();
  // we want to print at least an empty line, thus do-while construct
  do
  {
    mParent->mStream << std::setw(kWidth);
    if (kManip)
    {
      mParent->mStream << kManip;
    }
    mParent->prettyPrint<kWidth>(current, mBuffer.end());
    mParent->mStream << std::endl;
  } while (current != mBuffer.end());
}// end of Terminal::SimpleProxy::~SimpleProxy
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Destructor
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
template<std::size_t kMiddlePoint,
         iomanip     kLeftManip,
         iomanip     kRightManip>
Terminal<kWidth, kBufSize>::SplitProxy<kMiddlePoint, kLeftManip, kRightManip>::~SplitProxy()
{
  // correctly handle moved object, we don't want to duplicate the output
  if (!mParent)
  {
    return;
  }
  const_iterator column1 = mLeftBuffer.begin();
  const_iterator column2 = mRightBuffer.begin();
  // we want to print at least an empty line, thus do-while construct
  do
  {
    mParent->mStream << std::setw(kMiddlePoint - 1);
    if (kLeftManip)
    {
      mParent->mStream << kLeftManip;
    }
    mParent->prettyPrint<kMiddlePoint - 1>(column1, mLeftBuffer.end());
    mParent->mStream << "   ";
    mParent->mStream << std::setw(kWidth - kMiddlePoint - 2);
    if (kRightManip)
    {
      mParent->mStream << kRightManip;
    }
    mParent->prettyPrint<kWidth - kMiddlePoint - 2>(column2, mRightBuffer.end());
    mParent->mStream << std::endl;
  } while (column1 != mLeftBuffer.end() || column2 != mRightBuffer.end());
}// end of Terminal::SplitProxy::~SplitProxy
//----------------------------------------------------------------------------------------------------------------------

#endif /* TERMINAL_H */

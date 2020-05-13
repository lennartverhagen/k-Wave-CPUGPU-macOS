/**
 * @file Hdf5Io/Hdf5Error.cpp
 *
 * @brief Utilities to assemble error message by traversing HDF5 error stack
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

#include <Hdf5Io/Hdf5Error.h>

#include <sstream>
#include <vector>

#include <hdf5.h>

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Declarations ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Structure storing information obtained from H5E_error2_t
struct Hdf5ErrorItem
{
  /// Major error ID – HDF5 ID describing error category
  hid_t majorId;
  /// Minor error ID – HDF5 ID explaining the error cause
  hid_t minorId;
  /// Error detail – additional error description
  std::string detail;
};// end of Hdf5ErrorItem
//----------------------------------------------------------------------------------------------------------------------

/// Class for storing up to two errors from the HDF5 error stack with delayed evaluation
class Hdf5ErrorList
{
  public:
    /// Constructor initializing no errors
    Hdf5ErrorList() : mTop{H5E_DEFAULT, H5E_DEFAULT, ""}, mCause{H5E_DEFAULT, H5E_DEFAULT, ""}, mTruncated(false) {}

    /**
     * Method to store the error on the top of the stack
     * @param [in] source – Source of the error
     */
    void setTopError(const H5E_error_t& source);
    /**
     * Method to store the second error on the stack (just below the top)
     * @param [in] source – Source of the error
     */
    void setReasonError(const H5E_error_t& source);
    /// Method setting a flag that there were more errors
    void markTruncated() { mTruncated = true; }

    /**
     * Method composing the final message using the obtained information
     * @param [in, out] target – Where to print the message
     */
    void formatMessage(std::ostream& target);

  private:
    /// Error from the stack top
    Hdf5ErrorItem mTop;
    /// The second error on the stack
    Hdf5ErrorItem mCause;
    /// Flag specifying that further errors were present on the stack
    bool mTruncated;
};// end of Hdf5ErrorList
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Method to store the error on the top of the stack
 */
void Hdf5ErrorList::setTopError(const H5E_error2_t& source)
{
  mTop.majorId = source.maj_num;
  mTop.minorId = source.min_num;
  mTop.detail  = std::string(source.desc);
}// end of Hdf5ErrorList::setTopError
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to store the second error on the stack (just below the top)
 */
void Hdf5ErrorList::setReasonError(const H5E_error2_t& source)
{
  mCause.majorId = source.maj_num;
  mCause.minorId = source.min_num;
  mCause.detail  = std::string(source.desc);
}// end of setReasonError
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method composing the final message using the obtained information
 */
void Hdf5ErrorList::formatMessage(std::ostream& target)
{
  std::vector<char> buffer;
  ssize_t msgLength = H5Eget_msg(mTop.minorId, nullptr, nullptr, 0);
  if (msgLength <= 0)
  {
    target << "<description assembly failed>";
    return;
  }
  buffer.resize(msgLength + 1);
  msgLength = H5Eget_msg(mTop.minorId, nullptr, buffer.data(), buffer.size());
  if (msgLength <= 0)
  {
    target << "<description assembly failed>";
    return;
  }
  target << buffer.data();
  if (mTop.detail.length())
  {
    target << ": " << mTop.detail;
  }

  // the cause
  if (mCause.majorId != H5E_DEFAULT)
  {
    target << " (caused by ";
    msgLength = H5Eget_msg(mCause.minorId, nullptr, nullptr, 0);
    if (msgLength <= 0)
    {
      target << "<description assembly failed>" << (mTruncated ? "...)" : ")");
      return;
    }
    buffer.resize(msgLength + 1);
    msgLength = H5Eget_msg(mCause.minorId, nullptr, buffer.data(), buffer.size());
    if (msgLength <= 0)
    {
      target << "<description assembly failed>" << (mTruncated ? "...)" : ")");
      return;
    }
    target << buffer.data();
    if (mCause.detail.length())
    {
      target << ": " << mCause.detail;
    }
    target << (mTruncated ? "...)" : ")");
  }
}// end of formatMessage
//----------------------------------------------------------------------------------------------------------------------

extern "C"
{
  /**
   * @brief  HDF5 error stack traversal function filling up the passed Hdf5ErrorList
   * @param [in] n                – Level in the error
   * @param [in] errorDescription – Error description
   * @param [out] list            – List  of errors
   * @return error 
   */
  herr_t hdf5Walker(unsigned           n,
                    const H5E_error_t* errorDescription,
                    Hdf5ErrorList*     list)
  {
    try
    {
      if (n > 2)
      {
        // we have nothing to do with rest of the errors
        list->markTruncated();
      }
      else if (n == 0)
      {
        list->setTopError(*errorDescription);
      }
      else if (n == 1)
      {
        list->setReasonError(*errorDescription);
      }
    }
    catch (...)
    {
      return -1;
    }
    return 0;
  }// end of hdf5Walker
  //--------------------------------------------------------------------------------------------------------------------
}// end of extern "C"

/**
 * @brief Function returning an error string assembled by traversing HDF5 error stack
 */
std::string getHdf5ErrorString(const std::string& message)
{
  Hdf5ErrorList list;
  std::stringstream output;
  if (message.length())
  {
    output << message << ": ";
  }

  if (H5Ewalk(H5E_DEFAULT, H5E_WALK_UPWARD, reinterpret_cast<H5E_walk_t>(hdf5Walker), &list) < 0)
  {
    output << "<description assembly failed>";
  }
  else
  {
    list.formatMessage(output);
  }
  return output.str();
}// end of getHdf5ErrorString
//----------------------------------------------------------------------------------------------------------------------

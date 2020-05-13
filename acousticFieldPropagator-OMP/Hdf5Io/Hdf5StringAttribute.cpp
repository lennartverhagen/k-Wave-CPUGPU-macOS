/**
 * @file Hdf5Io/Hdf5StringAttribute.cpp
 *
 * @brief Maintaining HDF5 string attributes
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
 * Created: 2020-02-17 18:38\n
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

#include <Hdf5Io/Hdf5StringAttribute.h>

#include <algorithm>
#include <sstream>

#include <Hdf5Io/Hdf5Error.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Method to check if the attribute is present in the file already
 */
bool Hdf5StringAttribute::exists() const
{
  return mAttributeDesc >= 0;
}// end of Hdf5StringAttribute::exists
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Class for HDF5 string attribute manipulation
 */
std::string Hdf5StringAttribute::read() const
{
  if (!exists())
  {
    throw std::runtime_error("Cannot read non-existing attribute " + mAttributeName);
  }
  // get the current length
  Hdf5Id<H5Tclose> dataType = H5Aget_type(mAttributeDesc);
  if (dataType < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to obtain a length of the attribute " + mAttributeName));
  }
  size_t length = H5Tget_size(dataType);

  // Fixed-length strings in HDF5 may differ in a) padding / termination (see H5Tset_strpad) and b) character set.
  // In any case, it should be safe to read the string in the format that has been stored, in which case no conversion
  // is necessary and the reported length should suffice. Can deal with space padding afterwards.
  std::string result(length, '\0');
  // since C++11, the std::basic_string is guaranteed to use contiguous memory, so accessing the internal buffer is safe
  herr_t status = H5Aread(mAttributeDesc, dataType, &result[0]);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to read attribute " + mAttributeName));
  }
  // strip all spaces and null terminating characters from the end of the string
  result.erase(std::find_if(result.rbegin(), result.rend(), [](char ch) { return ch != '\0' && ch != ' '; }).base(),
               result.end());
  return result;
}// end of Hdf5StringAttribute::read
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method writing the attribute
 */
void Hdf5StringAttribute::write(const std::string& content)
{
  // if the attribute is present in the file already, fail
  // this behaviour can be easily changed by closing the attribute and calling
  // H5Adelete(mLocation, mAttributeName.c_str()) before proceeding, but not needed right now
  if (exists())
  {
    throw std::runtime_error("Attribute " + mAttributeName + " present already");
  }

  // we always write a string in the predefined H5T_C_S1 format
  Hdf5Id<H5Tclose> type = H5Tcopy(H5T_C_S1);
  if (type < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to create type for attribute " + mAttributeName));
  }
  herr_t status = H5Tset_size(type, content.size() + 1);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to set type size for attribute " + mAttributeName));
  }

  // first, we need to create a dataspace and attribute in the file
  Hdf5Id<H5Sclose> space = H5Screate(H5S_SCALAR);
  if (space < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to create space for attribute " + mAttributeName));
  }

  mAttributeDesc = H5Acreate(mLocation, mAttributeName.c_str(), type, space, H5P_DEFAULT, H5P_DEFAULT);

  // write the attribute
  status = H5Awrite(mAttributeDesc, type, content.c_str());
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to write the attribute " + mAttributeName));
  }
}//end of write
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Main class constructor
 */
Hdf5StringAttribute::Hdf5StringAttribute(hid_t       location,
                                         const char* name)
  : mLocation(location),
    mAttributeName(name)
{
  htri_t exists = H5Aexists(location, name);
  if (exists < 0)
  {
    std::stringstream ss;
    ss << "Failed to check existence of " << name << " attribute";
    throw std::runtime_error(getHdf5ErrorString(ss.str()));
  }
  if (exists)
  {
    // proceed with opening up the attribute and the corresponding dataspace
    mAttributeDesc = H5Aopen(location, name, H5P_DEFAULT);
    if (mAttributeDesc < 0)
    {
      throw std::runtime_error(getHdf5ErrorString("Failed to open attribute " + mAttributeName));
    }

    // check whether the data type is actually a fixed-length string
    Hdf5Id<H5Tclose> dataType = H5Aget_type(mAttributeDesc);
    if (dataType < 0)
    {
      throw std::runtime_error(getHdf5ErrorString("Failed to obtain a type of the attribute " + mAttributeName));
    }
    if (H5Tget_class(dataType) != H5T_STRING)
    {
      throw std::runtime_error("The attribute " + mAttributeName + " is not a string");
    }
    if (H5Tis_variable_str(dataType) > 0)
    {
      throw std::runtime_error("The attribute " + mAttributeName + " has variable length, which is unsupported");
    }
  }
}// end of Hdf5StringAttribute::Hdf5StringAttribute
//----------------------------------------------------------------------------------------------------------------------

/**
 * @file Hdf5Io/Hdf5StringAttribute.h
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

#ifndef HDF5STRINGATTRIBUTE_H
#define HDF5STRINGATTRIBUTE_H

#include <hdf5.h>

#include <Hdf5Io/Hdf5Dataset.h>
#include <Hdf5Io/Hdf5File.h>
#include <Hdf5Io/Hdf5Id.h>

/**
 * @brief Class for HDF5 string attribute manipulation
 *
 * This class can be used to read and write HDF5 attributes in a fixed-length string format. It is possible to use
 * either `Hdf5File` to work with attributes attached to a root group ("/", file attributes) or to specify a
 * `Hdf5Dataset` and work with dataset attributes instead. Passing absolute paths as names is also possible but not
 * recommended. Overwriting attributes is not supported.
 */
class Hdf5StringAttribute
{
  public:
    /**
     * @brief Constructs a handle to a file attribute
     *
     * One of the public interfaces. For a description please look at the
     * `Hdf5StringAttribute::Hdf5StringAttribute(hid_t location, const char* name)` constructor.
     *
     * @param[in] file – HDF5 file to work with
     * @param[in] name – Name of the attribute
     * @throws std::runtime_error if the initialization fails
     */
    Hdf5StringAttribute(Hdf5File& file, const char* name) : Hdf5StringAttribute(file.mFileDesc, name) {}

    /**
     * @brief Constructs a handle to a dataset attribute
     *
     * One of the public interfaces. For a description please look at the
     * `Hdf5StringAttribute::Hdf5StringAttribute(hid_t location, const char* name)` constructor.
     *
     * @param[in] dataset – HDF5 dataset to work with
     * @param[in] name    – Name of the attribute
     * @throws std::runtime_error if the initialization fails
     */
    Hdf5StringAttribute(Hdf5Dataset& dataset, const char* name) : Hdf5StringAttribute(dataset.mDatasetDesc, name) {}

    /**
     * @brief Method to check if the attribute is present in the file already
     * @returns true, if the represented attribute is already present in the file
     */
    bool exists() const;

    /**
     * @brief Method to read attribute's content
     *
     * The returned string may be right-justified and the trailing spaces removed. This is done since the storage
     * format allows for space padded strings.
     *
     * @returns The string stored in the attribute
     * @throws std::runtime_error if the reading fails
     */
    std::string read() const;

    /**
     * @brief Method writing the attribute
     *
     * Note that the attribute must not exist before writing or the method will throw an exception. Always stored
     * as a zero-terminated string.
     *
     * @param[in] content The content to write as an attribute
     * @throws std::runtime_error if the writing fails
     */
    void write(const std::string& content);

  private:
    /**
     * @brief Main class constructor
     *
     * Constructs and object representing the HDF5 attribute. If the specified attribute exists in the file already,
     * it is checked whether it is in a supported format. Such attribute can be read using the `read()` method. If the
     * attribute does not exist yet the constructor does no additional work. Upon calling the
     * `write(const std::string&)` method, the attribute is created and written into. After the write operation it can
     * be read as any other attribute that was present in the file previously.
     *
     * @param[in] location – A valid HDF5 handle pointing to a location where the attribute is (or should be)
     * @param[in] name     – Name of the attribute (or path according to HDF5 specification)
     * @throws std::runtime_error if underlying operations fail or the attribute is not in a supported string format
     */
    Hdf5StringAttribute(hid_t location, const char* name);

    /// Location, for lazy attribute creation
    // intentionally not using Hdf5Id wrapper for location since we don't own this object
    hid_t mLocation;
    /// Attribute name, for error messages
    std::string mAttributeName;
    /// Attribute descriptor (id)
    Hdf5Id<H5Aclose> mAttributeDesc;
};// end of Hdf5StringAttribute
//----------------------------------------------------------------------------------------------------------------------

#endif // HDF5STRINGATTRIBUTE_H

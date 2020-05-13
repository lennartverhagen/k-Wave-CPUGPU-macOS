/**
 * @file Hdf5Io/Hdf5File.h
 *
 * @brief Base class for HDF5 files
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

#ifndef HDF5_FILE_H
#define HDF5_FILE_H

#include <hdf5.h>

#include <Hdf5Io/Hdf5Id.h>

/**
 * @brief RAII wrapper for HDF5 file
 *
 * Class that takes care of closing HDF5 file when it's no longer needed. It's a wrapper for C bindings of a HDF5
 * library.
 */
class Hdf5File
{
    /// Dataset needs to access the file descriptor on creation
    friend class Hdf5Dataset;
    /// Attribute needs to access the file descriptor on creation
    friend class Hdf5StringAttribute;

  public:
    /**
     * @brief Method to close the underlying file explicitly
     *
     * Any operations on a closed file will most likely fail.
     */
    void close() {  mFileDesc = H5I_BADID; }
    /**
     * @brief Method to check if the file is open
     * @returns true, if the Hdf5File is open, false otherwise
     */
    bool isOpen() { return mFileDesc >= 0; }

  protected:
    /**
     * @brief Constructor
     *
     * Constructor is protected; it is expected to inherit this class and create or open the file in the derived class.
     */
    Hdf5File() {}

    /// Copy constructor not allowed.
    Hdf5File(const Hdf5File&)            = delete;
    /// Operator = not allowed.
    Hdf5File& operator=(const Hdf5File&) = delete;
    /// Move constructor not allowed.
    Hdf5File(Hdf5File&& orig)            = delete;

    /// HDF5 file descriptor (id)
    Hdf5Id<H5Fclose> mFileDesc;
};// end of Hdf5File
//----------------------------------------------------------------------------------------------------------------------

#endif /* HDF5_FILE_H */

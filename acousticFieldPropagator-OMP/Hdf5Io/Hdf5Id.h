/**
 * @file Hdf5Io/Hdf5Id.h
 *
 * @brief Basic templated RAII wrapper for hid_t
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
 * Created: 2020-02-07 15:31\n
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

#ifndef HDF5ID_H
#define HDF5ID_H

#include <hdf5.h>

/**
 * @brief Class providing a basic RAII wrapper on top of hid_t
 *
 * `hid_t` is a datatype used by HDF5 library to store object handles. These handles need to be released after their
 * lifetime to prevent resource leaks. When implementing classes that *own* a `hid_t` handle, this wrapper should be
 * used to ensure proper cleanup on all return paths. It can be also used as a scoped RAII wrapper for resources
 * held inside functions.
 *
 * The ownership is unique, i.e. only a single `Hdf5Id` object shall own a single `hid_t`. To partially overcome this
 * issue, Hdf5Id object can be only moved and move assigned. Please be aware that bugs involving premature resource
 * release can still occur since there is an implicit conversion to the underlying `hid_t` datatype. **Do not construct
 * or move-assign Hdf5Id object from an lvalue Hdf5Id object reference!**
 *
 * To use hid_t handles in functions or methods that merely *use them* and do *not* take their ownership, use `hid_t`
 * datatype directly. Such copy should not outlive the owning Hdf5Id object since the handle will be invalidated on
 * Hdf5Id destruction.
 *
 * @tparam kDeleter – Function that releases the resources held by stored hid_t
 */
template<herr_t (*kDeleter)(hid_t)>
class Hdf5Id
{
  public:
    /**
     * @brief Constructor
     *
     * Constructs a Hdf5Id object that doesn't hold any HDF5 resource.
     */
    Hdf5Id() : mId(H5I_UNINIT) {}

    /**
     * @brief Initializing constructor
     *
     * Constructs a Hdf5Id object that owns a given resource. Do not pass lvalue Hdf5Id references nor handles that are
     * already owned by other Hdf5Id wrappers!
     *
     * @param id HDF5 resource to gain ownership of
     */
    Hdf5Id(hid_t id) : mId(id) {}

    /// Copy constructor not allowed
    Hdf5Id(const Hdf5Id&)            = delete;
    /// Operator = not allowed
    Hdf5Id& operator=(const Hdf5Id&) = delete;

    // make Hdf5Id movable
    /**
     * @brief Move constructor
     *
     * Safely passes the ownership of the provided Hdf5Id object to the newly constructed one.
     *
     * @param[in] orig – Original Hdf5Id object
     */
    Hdf5Id(Hdf5Id&& orig);
    /**
     * @brief Move assignment operator
     *
     * Safely passes the ownership of the source Hdf5Id object. Previously held resource is released, if applicable.
     *
     * @param [in] orig – Original Hdf5Id object
     * @return
     */
    Hdf5Id& operator=(Hdf5Id&& orig);

    /**
     * @brief Implicit conversion to the underlying type
     *
     * To allow assignment of `hid_t` directly and also to allow implicit conversions for passing into functions that
     * accept `hid_t` directly.
     */
    operator hid_t() const { return mId; }

    /**
     * @brief Destructor
     *
     * Releases the owned resource, if applicable.
     */
    ~Hdf5Id();

  private:
    hid_t mId;
};// end of Hdf5Id
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Template methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 *@brief Move constructor
 */
template<herr_t (*kDeleter)(hid_t)>
Hdf5Id<kDeleter>::Hdf5Id(Hdf5Id<kDeleter>&& orig)
{
  mId      = orig.mId;
  orig.mId = H5I_UNINIT;
}// end of Hdf5Id::Hdf5Id
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Operator =
 */
template<herr_t (*kDeleter)(hid_t)>
Hdf5Id<kDeleter>& Hdf5Id<kDeleter>::operator=(Hdf5Id<kDeleter>&& orig)
{
  if (mId >= 0)
  {
    kDeleter(mId);
  }
  mId      = orig.mId;
  orig.mId = H5I_UNINIT;

  return *this;
}// end of Hdf5Id::operator=
//----------------------------------------------------------------------------------------------------------------------

/**
 *@brief  Destructor
 */
template<herr_t (*kDeleter)(hid_t)>
Hdf5Id<kDeleter>::~Hdf5Id()
{
  if (mId >= 0)
  {
    kDeleter(mId);
  }
}// end of Hdf5Id::~Hdf5Id
//----------------------------------------------------------------------------------------------------------------------

#endif // HDF5ID_H

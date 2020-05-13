/**
 * @file Hdf5Io/Hdf5Io.h
 *
 * @brief Input and output HDF5 file specializations based on Hdf5File
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

#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <sstream>

#include <hdf5.h>

#include <Hdf5Io/Hdf5Dataset.h>
#include <Hdf5Io/Hdf5Datatype.h>
#include <Hdf5Io/Hdf5Error.h>
#include <Hdf5Io/Hdf5File.h>
#include <Types/FftMatrix.h>
#include <Types/Parameters.h>
#include <Utils/Stopwatch.h>

/**
 * @brief Class representing opened HDF5 file (for input)
 */
class Hdf5Input : public Hdf5File
{
  public:
    /**
     * @brief Constructor
     *
     * Opens a HDF5 file with a given filename.
     *
     * @param[in] filename – Name of the file to open
     * @throws std::runtime_error if opening the file fails
     */
    Hdf5Input(const char* filename);

    /**
     * @brief Reads a scalar value from the file
     *
     * Since the input file is expected to be generated using Matlab, the scalars are expected to be stored as matrices
     * ("simple" HDF5 dataspace) containing a single element.
     *
     * @tparam     Type   – Type of the value to read
     * @param[in]  name   – Name of the dataset
     * @param[out] target – Target variable to read the value into
     */
    template<typename Type>
    void readScalar(const char* name,
                    Type&       target);

    /**
     * @brief Reads a vector value from the file
     * @tparam        Type   – Type of the value to read
     * @param[in]     name   – Name of the dataset
     * @param[in,out] target – Vector-like container to fill, expected to be pre-allocated and determines the correct
     *                         size
     */
    template<typename Type>
    void readVector(const char* name,
                    Type&       target);

    /**
     * @brief Check whether the file contains attributes and if their content is compatible
     * @returns true if the attributes are present
     */
    bool checkAttributes();

    /**
     * @brief Method to read calculation parameters
     * @param[out] params – Loaded parameters from the file
     * @throws std::runtime_error if the file does not contain expected datasets in an expected format
     */
    void readParams(Parameters& params);

    /**
     * @brief Method to read a 3D matrix from the file
     *
     * Reads a matrix from the HDF5 file with a given dataset name to the target `FftMatrix`.
     *
     * @param[in]  name               – Name of the dataset
     * @param[out] target             – Output matrix to write the data into
     * @param[in]  imaginaryComponent – Specifies if the data are treated as imaginary components, false by default
     */
    void readMatrix(const char* name,
                    FftMatrix&  target,
                    bool        imaginaryComponent = false);

    /**
     * @brief Method to read a 3D complex matrix from the file
     *
     * Reads a matrix from the HDF5 file with a given dataset name to the target `FftMatrix`. The input is expected to
     * be a 4-dimensional matrix, where the dimensionality of the slowest-varying direction is exactly 2, representing
     * the real and imaginary components respectively.
     *
     * @param[in]  name   – Name of the dataset
     * @param[out] target – Output matrix to write the data into
     */
    void readComplexMatrix(const char* name,
                           FftMatrix&  target);
};// end of Hdf5Input
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reads a scalar value from the file
 */
template<typename Type>
void Hdf5Input::readScalar(const char* name,
                           Type&       target)
{
  Hdf5Dataset dataset(*this, name);
  if (dataset.elementCount() != 1)
  {
    std::stringstream ss;
    ss << "The dataset " << name << " is not a scalar";
    throw std::runtime_error(ss.str());
  }
  dataset.read(Hdf5Datatype<Type>::kType, H5S_ALL, &target);
}// end of Hdf5Input::readScalar
//----------------------------------------------------------------------------------------------------------------------

/**
 * Reads a vector value from the file
 */
template<typename Type>
void Hdf5Input::readVector(const char* name,
                           Type&       target)
{
  Hdf5Dataset dataset(*this, name);
  if (dataset.elementCount() != target.size())
  {
    std::stringstream ss;
    ss << "The dataset " << name << " is not a " << target.size() << "-element vector";
    throw std::runtime_error(ss.str());
  }
  dataset.read(Hdf5Datatype<typename Type::value_type>::kType, H5S_ALL, target.data());
}// end of Hdf5Input::readVector
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Class representing created HDF5 file (for output)
 */
class Hdf5Output : public Hdf5File
{
  public:
    /**
     * @brief Constructor
     *
     * Creates a new HDF5 file with a given filename.
     *
     * @param[in] filename – Name of the file to create
     * @throws std::runtime_error if creating the file fails
     */
    Hdf5Output(const char* filename);

    /**
     * @brief Writes the basic, statically known attributes to the output file
     */
    void writeBasicAttributes();

    /**
     * @brief Writes the time specified as a file attribute
     *
     * @param[in] name     – Name of the file attribute to create
     * @param[in] duration – Stopwatch duration to use
     */
    void writeTimeAttribute(const char*                              name,
                            const Stopwatch::steady_clock::duration& duration);

    /**
     * @brief Writes the memory consumption specified as a file attribute
     *
     * @param[in] name      – Name of the file attribute to create
     * @param[in] memoryMiB – Attribute content to format
     */
    void writeMemoryAttribute(const char* name,
                              size_t      memoryMiB);

    /**
     * @brief Flushes the buffers associated with the file to the disk
     */
    void flush();

    /**
     * @brief Method to write a 3D matrix to the file
     *
     * Writes a matrix from the source `FftMatrix` to the HDF5 file with a given dataset name.
     *
     * @param[in]  name               – Name of the dataset
     * @param[out] source             – Source matrix to read the data from
     * @param[in]  imaginaryComponent – Specifies if imaginary components should be stored, false (real) by default
     */
    void writeMatrix(const char*      name,
                     const FftMatrix& source,
                     bool             imaginaryComponent = false);

    /**
     * @brief Method to write a 3D sub-matrix to the file
     *
     * Writes a sub-matrix from the source `FftMatrix` to the HDF5 file with a given dataset name. Sub-matrix is a
     * selection of the original with the given size, starting from (0,0,0).
     *
     * @param[in]  name               – Name of the dataset
     * @param[out] source             – Source matrix to read the data from
     * @param[in]  size               – Size of the sub-matrix to store
     * @param[in]  imaginaryComponent – Specifies if imaginary components should be stored, false (real) by default
     */
    void writeSubMatrix(const char*      name,
                        const FftMatrix& source,
                        const Size3D&    size,
                        bool             imaginaryComponent = false);

    /**
     * @brief Method to write a 3D complex matrix to the file
     *
     * Writes a matrix from the source `FftMatrix` to the HDF5 file with a given dataset name. Complex values are stored
     * by creating an additional dimension in the slowest-varying direction. If the complex matrix is of the size
     * (x,y,z), the HDF5 file will contain a 4-dimensional dataset of size (2,x,y,z), where z is the fastest-varying
     * direction. (0,x,y,z) then corresponds to real, and (1,x,y,z) to imaginary components of the stored numbers,
     * assuming indices starting with 0.
     *
     * @param[in]  name   – Name of the dataset
     * @param[out] source – Source matrix to read the data from
     */
    void writeComplexMatrix(const char*      name,
                            const FftMatrix& source);

    /**
     * @brief Method to write a 3D complex sub-matrix to the file
     *
     * Writes a sub-matrix from the source `FftMatrix` to the HDF5 file with a given dataset name. Sub-matrix is a
     * selection of the original with the given size, starting from (0,0,0). Complex values are stored by creating an
     * additional dimension in the slowest-varying direction. If the complex matrix is of the size (x,y,z), the HDF5
     * file will contain a 4-dimensional dataset of size (2,x,y,z), where z is the fastest-varying direction.
     * (0,x,y,z) then corresponds to real, and (1,x,y,z) to imaginary components of the stored numbers, assuming indices
     * starting with 0.
     *
     * @param[in]  name   – Name of the dataset
     * @param[out] source – Source matrix to read the data from
     * @param[in]  size   – Size of the sub-matrix to store
     */
    void writeComplexSubMatrix(const char*      name,
                               const FftMatrix& source,
                               const Size3D&    size);
};// end of Hdf5Output
//----------------------------------------------------------------------------------------------------------------------

#endif /* HDF5_IO_H */

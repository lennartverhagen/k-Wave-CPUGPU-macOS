/**
 * @file Hdf5Io/Hdf5Io.cpp
 *
 * @brief Class providing a basic RAII wrapper on top of hid_t
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

#include <Hdf5Io/Hdf5Io.h>

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <limits>

#include <Hdf5Io/Hdf5MemSpace.h>
#include <Hdf5Io/Hdf5StringAttribute.h>
#include <Utils/Hostname.h>
#include <Version/Version.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor
 */
Hdf5Input::Hdf5Input(const char* filename)
{
  mFileDesc = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (mFileDesc < 0)
  {
    throw std::runtime_error(getHdf5ErrorString(std::string("Failed to open HDF5 file ") + filename));
  }
}// end of Hdf5Input::Hdf5Input
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Check whether the file contains attributes and if their content is compatible
 */
bool Hdf5Input::checkAttributes()
{
  // check the file type first
  Hdf5StringAttribute fileType(*this, "file_type");
  if (!fileType.exists())
  {
    return false;
  }

  std::string fileTypeString = fileType.read();
  if (fileTypeString != "afp_input")
  {
    throw std::runtime_error("Input file doesn't have a proper type, expected afp_input, got " + fileTypeString);
  }

  // now check the file version, read the numbers
  Hdf5StringAttribute majorVersion(*this, "major_version");
  Hdf5StringAttribute minorVersion(*this, "minor_version");
  if (!majorVersion.exists() || !minorVersion.exists())
  {
    return false;
  }

  std::size_t lastPosition;
  std::string versionString        = minorVersion.read();
  unsigned long minorVersionNumber = std::stoul(versionString, &lastPosition);
  if (lastPosition != versionString.length())
  {
    throw std::runtime_error("Failed to read minor version attribute content");
  }

  versionString                    = majorVersion.read();
  unsigned long majorVersionNumber = std::stoul(versionString, &lastPosition);
  if (lastPosition != versionString.length())
  {
    throw std::runtime_error("Failed to read major version attribute content");
  }

  // file version comparisons
  if (majorVersionNumber < kInputFileRequiredVersionMajor ||
      (majorVersionNumber == kInputFileRequiredVersionMajor && minorVersionNumber < kInputFileRequiredVersionMinor))
  {
    std::stringstream ss;
    ss << "Input file version " << majorVersionNumber << '.' << minorVersionNumber
       << " unsupported, minimum required version is " << kInputFileRequiredVersionMajor << '.'
       << kInputFileRequiredVersionMinor;
    throw std::runtime_error(ss.str());
  }
  if (majorVersionNumber > kInputFileSupportedVersionMajor ||
      (majorVersionNumber == kInputFileSupportedVersionMajor && minorVersionNumber > kInputFileSupportedVersionMinor))
  {
    std::stringstream ss;
    ss << "Input file version " << majorVersionNumber << '.' << minorVersionNumber
       << " unsupported, maximum supported version is " << kInputFileSupportedVersionMajor << '.'
       << kInputFileSupportedVersionMinor;
    throw std::runtime_error(ss.str());
  }
  return true;
}// end of Hdf5Input::checkAttributes
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to read calculation parameters
 */
void Hdf5Input::readParams(Parameters& params)
{
  // w0, c0, t, dx
  readScalar("w0", params.w0);
  readScalar("c0", params.c0);
  readScalar("t", params.t);
  readScalar("dx", params.dx);

  // sz_ex
  // perform basic static tests on the types
  static_assert(sizeof(std::size_t) == sizeof(unsigned long long),
                "size_t and unsigned long long have a different size and this implementation is malformed");
  static_assert(std::numeric_limits<std::size_t>::is_signed == false,
                "size_t is signed and this implementation is malformed");
  readVector("sz_ex", params.extended);
  // for some great reasons, the order of the dimension sizes is reversed in the hdf5 file; fix this
  std::swap(params.extended.x(), params.extended.z());

  // check whether the input is complex (source_in) or separated (amp_in + phase_in)
  if (H5Lexists(mFileDesc, "source_in", H5P_DEFAULT))
  {
    // check whether the amplitude and phase are present at the same time
    if (H5Lexists(mFileDesc, "amp_in", H5P_DEFAULT) || H5Lexists(mFileDesc, "phase_in", H5P_DEFAULT))
    {
      throw std::runtime_error("Both source_in and either of amp_in or phase_in found in the input file");
    }
    // get the size
    Hdf5Dataset sourceDataset(*this, "source_in");
    auto sourceSize = sourceDataset.size();
    std::fill_n(params.size.begin(), 3, 0);
    std::copy_n(sourceSize.rbegin(), std::min(std::size_t(3), sourceSize.size() - 1), params.size.rbegin());
    // set the flag
    params.complexInput = true;
  }
  else
  {
    // get the size
    Hdf5Dataset amplitudeDataset(*this, "amp_in");
    auto amplitudeSize = amplitudeDataset.size();
    std::fill_n(params.size.begin(), 3, 0);
    std::copy_n(amplitudeSize.rbegin(), std::min(std::size_t(3), amplitudeSize.size()), params.size.rbegin());

    // check if phase_in is scalar or not, and wether the size matches the amp_in
    Hdf5Dataset phaseDataset(*this, "phase_in");
    if (phaseDataset.elementCount() == 1)
    {
      params.phaseIsScalar = true;
      phaseDataset.read(Hdf5Datatype<float>::kType, H5S_ALL, &params.phase);
    }
    else
    {
      if (amplitudeSize != phaseDataset.size())
      {
        throw std::runtime_error("Sizes of amp_in and phase_in in the input file differ");
      }
      params.phaseIsScalar = false;
    }
    // set the flag
    params.complexInput = false;
  }
}// end of Hdf5Input::readParams
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to read a 3D matrix from the file
 */
void Hdf5Input::readMatrix(const char* name,
                           FftMatrix&  target,
                           bool        imaginaryComponent)
{
  // open up the dataset and check the rank
  Hdf5Dataset dataset(*this, name);
  if (dataset.rank() != 3)
  {
    std::stringstream ss;
    ss << "The dataset " << name << " is incompatible with the FftMatrix component format (wrong rank)";
    throw std::runtime_error(ss.str());
  }

  Hdf5MemSpace memSpace(target.size(), dataset.size());

  // decide where is the target
  float* data = imaginaryComponent ? target.imagData() : target.realData();
  dataset.read(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<void*>(data));
}// end of Hdf5Input::readMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to read a 3D complex matrix from the file
 */
void Hdf5Input::readComplexMatrix(const char* name,
                                  FftMatrix&  target)
{
  // open up the dataset and check the rank
  // also check the size in the slowest varying direction -- should be 2
  Hdf5Dataset dataset(*this, name);
  auto complexSize = dataset.size();
  if (complexSize.size() != 4)
  {
    std::stringstream ss;
    ss << "The dataset " << name << " is incompatible with FftMatrix complex format (wrong rank)";
    throw std::runtime_error(ss.str());
  }

  if (complexSize[0] != 2)
  {
    std::stringstream ss;
    ss << "The dataset " << name
       << " is incompatible with FftMatrix complex format (wrong size in the slowest varying dimension)";
    throw std::runtime_error(ss.str());
  }

  // create memory space
  std::array<hsize_t, 3> realSize;
  std::copy_n(complexSize.begin() + 1, 3, realSize.begin());
  Hdf5MemSpace memSpace(target.size(), realSize);

  // restrict to real and copy
  std::array<hsize_t, 4> start{0, 0, 0, 0};
  complexSize[0] = 1;
  dataset.select(start, complexSize);
  dataset.read(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<void*>(target.realData()));

  // restrict to imaginary and copy
  start[0] = 1;
  dataset.select(start, complexSize);
  dataset.read(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<void*>(target.imagData()));

}// end of Hdf5Input::readComplexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Constructor
 */
Hdf5Output::Hdf5Output(const char* filename)
{
  mFileDesc = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (mFileDesc < 0)
  {
    throw std::runtime_error(getHdf5ErrorString(std::string("Failed to create HDF5 file ") + filename));
  }
}// end of Hdf5Output::Hdf5Output
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Writes the basic, statically known attributes to the output file
 */
void Hdf5Output::writeBasicAttributes()
{
  // creation tool
  Hdf5StringAttribute(*this, "created_by").write(std::string("acousticFieldPropagator-OMP ") + kRevisionString);

  // format creation date, human readable local time
  std::time_t time = std::time(nullptr);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time), "%c %Z");
  Hdf5StringAttribute(*this, "creation_date").write(ss.str());

  // file type
  Hdf5StringAttribute(*this, "file_type").write("afp_output");

  // file versions
  Hdf5StringAttribute(*this, "major_version").write(std::to_string(kOutputFileVersionMajor));
  Hdf5StringAttribute(*this, "minor_version").write(std::to_string(kOutputFileVersionMinor));

  // hostname
  Hdf5StringAttribute(*this, "host_names").write(getHostname());
}// end of Hdf5Output::writeBasicAttributes
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Writes the time specified as a file attribute
 */
void Hdf5Output::writeTimeAttribute(const char*                              name,
                                    const Stopwatch::steady_clock::duration& duration)
{
  std::chrono::duration<double, std::milli> time(duration);
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << time.count() << " ms";
  Hdf5StringAttribute(*this, name).write(ss.str());
}// end of Hdf5Output::writeTimeAttribute
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Writes the memory consumption specified as a file attribute
 */
void Hdf5Output::writeMemoryAttribute(const char* name,
                                      size_t      memoryMiB)
{
  std::stringstream ss;
  ss << memoryMiB << " MiB";
  Hdf5StringAttribute(*this, name).write(ss.str());
}// end of Hdf5Output::writeMemoryAttribute
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Flushes the buffers associated with the file to the disk
 */
void Hdf5Output::flush()
{
  herr_t status = H5Fflush(mFileDesc, H5F_SCOPE_LOCAL);
  if (status < 0)
  {
    throw std::runtime_error(getHdf5ErrorString("Failed to perform a flush"));
  }
}// end of Hdf5Output::flush
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to write a 3D matrix to the file
 */
void Hdf5Output::writeMatrix(const char*      name,
                             const FftMatrix& source,
                             bool             imaginaryComponent)
{
  Hdf5Dataset dataset(*this, name, Hdf5Datatype<float>::kType, source.size());
  Hdf5MemSpace memSpace(source.size());
  const float* data = imaginaryComponent ? source.imagData() : source.realData();
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(data));
}// end of Hdf5Output::writeComplexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to write a 3D sub-matrix to the file
 */
void Hdf5Output::writeSubMatrix(const char*      name,
                                const FftMatrix& source,
                                const Size3D&    size,
                                bool             imaginaryComponent)
{
  Hdf5Dataset dataset(*this, name, Hdf5Datatype<float>::kType, size);
  Hdf5MemSpace memSpace(source.size(), size);
  const float* data = imaginaryComponent ? source.imagData() : source.realData();
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(data));
}// end of Hdf5Output::writeSubMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to write a 3D complex matrix to the file
 */
void Hdf5Output::writeComplexMatrix(const char*      name,
                                    const FftMatrix& source)
{
  // get the complex size of the dataset first
  std::array<std::size_t, 4> start{0, 0, 0, 0};
  std::array<std::size_t, 4> complexSize;
  complexSize[0] = 2;
  std::copy_n(source.size().begin(), 3, complexSize.begin() + 1);

  // create a dataset and memspace
  Hdf5Dataset dataset(*this, name, Hdf5Datatype<float>::kType, complexSize);
  Hdf5MemSpace memSpace(source.size());

  // write real component
  complexSize[0] = 1;
  dataset.select(start, complexSize);
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(source.realData()));

  // write imaginary component
  start[0] = 1;
  dataset.select(start, complexSize);
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(source.imagData()));
}// end of Hdf5Output::writeComplexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Method to write a 3D complex sub-matrix to the file
 */
void Hdf5Output::writeComplexSubMatrix(const char*      name,
                                       const FftMatrix& source,
                                       const Size3D&    size)
{
  // get the complex size of the dataset first
  std::array<std::size_t, 4> start{0, 0, 0, 0};
  std::array<std::size_t, 4> complexSize;
  complexSize[0] = 2;
  std::copy_n(size.begin(), 3, complexSize.begin() + 1);

  // create a dataset and memspace
  Hdf5Dataset dataset(*this, name, Hdf5Datatype<float>::kType, complexSize);
  Hdf5MemSpace memSpace(source.size(), size);

  // write real component
  complexSize[0] = 1;
  dataset.select(start, complexSize);
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(source.realData()));

  // write imaginary component
  start[0] = 1;
  dataset.select(start, complexSize);
  dataset.write(Hdf5Datatype<float>::kType, memSpace.space(), reinterpret_cast<const void*>(source.imagData()));
}// end of Hdf5Output::writeComplexSubMatrix
//----------------------------------------------------------------------------------------------------------------------

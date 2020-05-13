/**
 * @file Version/Version.h
 *
 * @brief Automatically generated version information
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
 * @version v1.0.0
 *
 * @date
 * Created: 2017-03-22 18:23\n
 * Last modified: 2020-02-20 09:49
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

#ifndef VERSION_H
#define VERSION_H

/// Major software version
extern const unsigned short kVersionMajor;
/// Minor software version
extern const unsigned short kVersionMinor;
/// Patch number
extern const unsigned short kVersionPatch;

/// Revision string
extern const char* kRevisionString;
/// Status of the working directory and staging area
extern const char* kRepositoryStatus;

/// Build date and time
extern const char* kBuildDateTime;

/// C++ compiler identification
extern const char* kCompilerIdentifier;

/// Output file major version
static constexpr unsigned short kOutputFileVersionMajor{1};
/// Output file minor version
static constexpr unsigned short kOutputFileVersionMinor{0};
/// File minimum major version required
static constexpr unsigned short kInputFileRequiredVersionMajor{1};
/// File minimum minor version required
static constexpr unsigned short kInputFileRequiredVersionMinor{0};
/// File maximum major version supported
static constexpr unsigned short kInputFileSupportedVersionMajor{1};
/// File maximum minor version supported
static constexpr unsigned short kInputFileSupportedVersionMinor{0};

#ifdef KWAVE_VERSION
/// kWave release version number
extern const char* kKWaveVersion;
#endif

#endif /* VERSION_H */

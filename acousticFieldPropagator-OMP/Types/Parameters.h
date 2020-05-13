/**
 * @file Types/Parameters.h
 *
 * @brief Simulation parameters passed to several routines
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

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <Types/TypedTuple.h>

/**
 * @brief Structure containing simulation parameters
 */
struct Parameters
{
    /// Angular speed [rad/s]
    float w0;
    /// Acoustic velocity in the media [m/s]
    float c0;
    /// Time point to calculate pressure field at [s]
    float t;
    /// Spatial grid spacing [m]
    float dx;

    /// Size of the computational grid
    TypedTuple<std::size_t, 3> size;
    /// Extended size of the computational grid (to tackle domain periodicity)
    TypedTuple<std::size_t, 3> extended;

    /// Flag if the phase is a scalar
    bool  phaseIsScalar;
    /// Scalar phase, valid if phaseIsScalar is true
    float phase;

    /// Flag whether the input is complex or needs preprocessing
    bool  complexInput;
};// end of Parameters
//----------------------------------------------------------------------------------------------------------------------

#endif /* PARAMETERS_H */

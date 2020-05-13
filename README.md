# k-Wave-CPUGPU-macOS

Copyright (C) 2012 - 2020 SC\@FIT Research Group,
Brno University of Technology, Brno, CZ.

This file is part of the C++ extension of the k-Wave Toolbox
(http://www.k-wave.org).

Modifications: macOS compatibility for OMP by Lennart Verhagen, Donders Institute, Radboud University, Nijmegen, the Netherlands


## Overview

k-Wave is an open source MATLAB toolbox designed for the time-domain simulation
of propagating acoustic waves in 1D, 2D, or 3D. The toolbox has a wide range of
functionality, but at its heart is an advanced numerical model that can account
for both linear or nonlinear wave propagation, an arbitrary distribution of
heterogeneous material parameters, and power law acoustic absorption.
See the k-Wave website (http://www.k-wave.org) for further details.

This project is a part of the k-Wave toolbox accelerating 2D/3D simulations
using an optimized C++ implementation to run small to moderate grid sizes (e.g.,
128x128 to 10,000x10,000 in 2D or 64x64x64 to 512x512x512 in 3D) on systems
with shared memory. 2D simulations can be carried out in both normal and
axisymmetric coordinate systems.


## macOS compatibility

The c++ CPU optimised code supporting OpenMP has been modified from the linux distribution to be compatible with macOS. These modifications have only been tested on macOS Catalina 10.15.

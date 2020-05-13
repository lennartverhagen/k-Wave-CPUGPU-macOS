/**
 * @file      SolverCudaKernels.cuh
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for all cuda kernels used in the GPU implementation of the k-space solver.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 March     2013, 13:10 (created) \n
 *            11 February  2020, 16:14 (revised)
 *
 * @copyright Copyright (C) 2013 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#ifndef SOLVER_CUDA_KERNELS_H
#define	SOLVER_CUDA_KERNELS_H

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>
#include <MatrixClasses/CufftComplexMatrix.h>

#include <Containers/MatrixContainer.h>
#include <Utils/DimensionSizes.h>

#include <Parameters/Parameters.h>
#include <Parameters/CudaParameters.h>

/**
 * @class   SolverCudaKernels
 * @brief   Static class with all cuda kernels used in the solver.
 * @details Static class with all cuda kernels used in the solver. This class cannot be instantiated. The on reason why
 *          a namespace is not used it the template capabilities of the class that allows to remove deep decision trees.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 * @tparam bOnAScalarFlag      - Is nonlinearity homogeneous?
 * @tparam c0ScalarFlag        - Is sound speed homogenous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
 */
template<Parameters::SimulationDimension simulationDimension = Parameters::SimulationDimension::k2D,
         bool                            rho0ScalarFlag      = true,
         bool                            bOnAScalarFlag      = true,
         bool                            c0ScalarFlag        = true,
         bool                            alphaCoefScalarFlag = true>
class SolverCudaKernels
{
  public:
    /// Default constructor not allowed (static class).
    SolverCudaKernels() = delete;
    /// Copy constructor not allowed (static class).
    SolverCudaKernels(SolverCudaKernels&) = delete;
    /// Destructor not allowed (static class).
    ~SolverCudaKernels() = delete;

    /**
     * @brief   Get the cuda architecture the code was compiled for.
     * @details It is done by calling a kernel that reads a variable set by nvcc compiler.
     *
     * @return  The cuda code version the code was compiled for.
     */
    static int getCudaCodeVersion();

    //----------------------------------------- Compute pressure gradient --------------------------------------------//
    /**
     * @brief Compute spectral part of pressure gradient in between FFTs.
     *
     * <b> Matlab code: </b> \code
     *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
     *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
     *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
     * \endcode
     */
    static void computePressureGradient();

    //---------------------------------------------- Compute velocity ------------------------------------------------//
    /**
     * @brief Compute acoustic velocity for a uniform grid.
     *
     * <b> Matlab code: </b> \code
     *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX)
     *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY)
     *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ)
     * \endcode
     */
    static void computeVelocityUniform();

    /**
     * @brief Compute acoustic velocity for homogenous medium and nonuniform grid.
     *
     * <b> Matlab code: </b> \code
     *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx)  ...
     *                  - dt .* rho0_sgx_inv .* dxudxnSgx.* real(ifftX));
     *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) ...
     *                  - dt .* rho0_sgy_inv .* dyudynSgy.* real(ifftY);
     *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz)
     *                  - dt .* rho0_sgz_inv .* dzudznSgz.* real(ifftZ);
     *\endcode
     */
    static void computeVelocityHomogeneousNonuniform();

    /// Compute the velocity shift in Fourier space over x direction.
    static void computeVelocityShiftInX();
    /// Compute the velocity shift in Fourier space over y direction.
    static void computeVelocityShiftInY();
    /// Compute the velocity shift in Fourier space over z direction.
    static void computeVelocityShiftInZ();

    //----------------------------------------- Compute velocity gradient --------------------------------------------//
    /**
     * @brief Compute spatial part of the velocity gradient in between FFTs on uniform grid.
     *
     * <b> Matlab code: </b> \code
     *  bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx));
     *  bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy));
     *  bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz));
     * \endcode
     */
    static void computeVelocityGradient();

    /// Shift gradient of acoustic velocity on non-uniform grid.
    static void computeVelocityGradientShiftNonuniform();

    //---------------------------------------------- Compute density -------------------------------------------------//
    /**
     * @brief Compute acoustic density for non-linear case.
     *
     * <b>Matlab code:</b> \code
     *  rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
     *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
     *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
     *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
     * \endcode
     */
    static void computeDensityNonlinear();

    /**
     * @brief Compute acoustic density for linear case.
     *
     * <b>Matlab code:</b> \code
     *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
     *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
     *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
     * \endcode
     */
    static void computeDensityLinear();

    //----------------------------------------- Compute nonlinear pressure -------------------------------------------//
    /**
     * @brief Sum sub-terms for new pressure in linear lossless case.
     *
     * <b>Matlab code:</b> \code
     *  % calculate p using a nonlinear adiabatic equation of state
     *  p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
     * \endcode
     */
    static void sumPressureNonlinearLossless();

    /**
     * @brief Compute three temporary sums in the new pressure formula in non-linear power law case.
     *
     * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
     * @param [out] nonlinearTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz);
     * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
     */
    static void computePressureTermsNonlinearPowerLaw(RealMatrix& densitySum,
                                                      RealMatrix& nonlinearTerm,
                                                      RealMatrix& velocityGradientSum);
    /**
     * @brief Compute absorbing term with abosrbNabla1 and absorbNabla2.
     *
     * <b>Matlab code:</b> \code
     *  fftPart1 = absorbNabla1 .* fftPart1;
     *  fftPart2 = absorbNabla2 .* fftPart2;
     * \endcode
     */
    static void computeAbsorbtionTerm();
    /**
     * @brief Sum sub-terms to compute new pressure in non-linear power law case.
     *        The output is stored into the pressure matrix.
     *
     * @param [in] nonlinearTerm - Nonlinear term
     * @param [in] absorbTauTerm - Absorb tau term from the pressure eq.
     * @param [in] absorbEtaTerm - Absorb eta term from the pressure eq.
     *
     * <b>Matlab code:</b> \code
     *  % calculate p using a nonlinear absorbing equation of state
     *  p = c0.^2 .* (...
     *                nonlinearTerm ...
     *                + absorb_tau .* absorbTauTerm...
     *                - absorb_eta .* absorbEtaTerm...
     *                );
     * \endcode
     */
    static void sumPressureTermsNonlinearPowerLaw(const RealMatrix& nonlinearTerm,
                                                  const RealMatrix& absorbTauTerm,
                                                  const RealMatrix& absorbEtaTerm);

    /**
     * @brief Sum sub-terms to compute new pressure in  nonlinear stokes case.
     *
     * <b>Matlab code:</b> \code
     *  p = c0.^2 .* ( ...
     *      (rhox + rhoy + rhoz) ...
     *       + absorb_tau .* rho0 .* (duxdx + duydy) ...
     *       + medium.BonA .* (rhox + rhoy).^2 ./ (2 .* rho0));
     * \endcode
     */
    static void sumPressureNonlinearStokes();

    //----------------------------------------- Compute nonlinear pressure -------------------------------------------//
    /**
     * @brief Sum sub-terms for new pressure in linear lossless case.
     *
     * <b>Matlab code:</b> \code
     *  % calculate p using a linear adiabatic equation of state
     *  p = c0.^2 .* (rhox + rhoy + rhoz);
     * \endcode
     *
     */
    static void sumPressureLinearLossless();

    /**
     * @brief Compute two temporary sums in the new pressure formula for linear power law case.
     *
     * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
     * @param [out] velocityGradientSum - rho0 * (duxdx + duydy + duzdz);
     */
    static void computePressureTermsLinearPowerLaw(RealMatrix& densitySum,
                                                   RealMatrix& velocityGradientSum);
    /**
     * @brief Sum sub-terms to compute new pressure in linear power law case.
     *        The output is stored into the pressure matrix.
     *
     * @param [in] absorbTauTerm - Absorb tau term from the pressure eq.
     * @param [in] absorbEtaTerm - Absorb tau term from the pressure eq.
     * @param [in] densitySum    - Sum of acoustic density.
     *
     * <b>Matlab code:</b> \code
     *  % calculate p using a nonlinear absorbing equation of state
     *  p = c0.^2 .* (...
     *                densitySum
     *                + absorb_tau .* absorbTauTerm...
     *                - absorb_eta .* absorbEtaTerm...
     *                );
     * \endcode
     */
    static void sumPressureTermsLinearPowerLaw(const RealMatrix& absorbTauTerm,
                                               const RealMatrix& absorbEtaTerm,
                                               const RealMatrix& densitySum);

    /**
     * @brief Sum sub-terms to compute new pressure in linear stokes case.
     *
     * <b>Matlab code:</b> \code
     *  p = c.^2 .* ((rhox + rhoy + rhoz) + absorb_tau .* rho0 .* (duxdx + duydy + duzdz));
     * \endcode
     */
    static void sumPressureLinearStokes();

    //-------------------------------------------------- Sources -----------------------------------------------------//
    /**
     * @brief Add in pressure source term.
     * @param [in] container       - Container with all matrices.
     */
    static void addPressureSource(const MatrixContainer& container);
    /**
     * @brief Add transducer data source to velocity x component.
     * @param [in] container - Container with all matrices.
     */
    static void addTransducerSource(const MatrixContainer& container);
    /**
     * @brief Add in velocity source terms.
     *
     * @param [in, out] velocity       - Velocity matrix to update.
     * @param [in] velocitySourceInput - Source input to add.
     * @param [in] velocitySourceIndex - Source geometry index matrix.
     */
    static void addVelocitySource(RealMatrix&        velocity,
                                  const RealMatrix&  velocitySourceInput,
                                  const IndexMatrix& velocitySourceIndex);

    /**
     * @brief Add scaled pressure source to acoustic density, 3D case.
     * @tparam simulationDimension - Dimensionality of the simulation.
     * @param [in] scaledSource    - Scaled source.
     */
    static void addPressureScaledSource(const RealMatrix& scalingSource);
    /**
     * @brief Add scaled velocity source to acoustic density.
     * @param [in, out] velocity     - Velocity matrix to update.
     * @param [in]      scaledSource - Scaled source.
     */
    static void addVelocityScaledSource(RealMatrix&        velocity,
                                        const RealMatrix&  scalingSource);

    /**
     * @brief Insert source signal into scaling matrix.
     *
     * @param [out] scaledSource - Temporary matrix to insert the source into before scaling.
     * @param [in]  sourceInput  - Source input signal.
     * @param [in]  sourceIndex  - Source geometry.
     * @param [in]  manyFlag     - Number of time series in the source input.
     */
    static void insertSourceIntoScalingMatrix(RealMatrix&        scaledSource,
                                              const RealMatrix&  sourceInput,
                                              const IndexMatrix& sourceIndex,
                                              const size_t       manyFlag);
    /**
     * @brief Compute source gradient.
     * @param [in, out] sourceSpectrum - Source spectrum.
     * @param [in]      sourceKappa    - Source kappa.
     */
    static void computeSourceGradient(CufftComplexMatrix& sourceSpectrum,
                                      const RealMatrix&   sourceKappa);


    /**
     * @brief Add initial pressure source to the pressure matrix and update density matrices.
     *
     * <b>Matlab code:</b> \code
     *  % add the initial pressure to rho as a mass source (3D code)
     *  p = source.p0;
     *  rhox = source.p0 ./ (3 .* c.^2);
     *  rhoy = source.p0 ./ (3 .* c.^2);
     *  rhoz = source.p0 ./ (3 .* c.^2);
     * \endcode
     */
    static void addInitialPressureSource();

    /**
     * @brief  Compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b> Matlab code: </b> \code
     *  ux_sgx = dt ./ rho0_sgx .* dxudxn_sgx .* ifft(ux_sgx);
     *  uy_sgy = dt ./ rho0_sgy .* dyudxn_sgy .* ifft(uy_sgy);
     *  uz_sgz = dt ./ rho0_sgz .* dzudzn_sgz .* ifft(uz_sgz);
     * \endcode
     */
    static void computeInitialVelocityHomogeneousNonuniform();
    /**
     * @brief Compute acoustic velocity for initial pressure problem, uniform grid.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b> Matlab code: </b> \code
     *  ux_sgx = dt ./ rho0_sgx .* ifft(ux_sgx);
     *  uy_sgy = dt ./ rho0_sgy .* ifft(uy_sgy);
     *  uz_sgz = dt ./ rho0_sgz .* ifft(uz_sgz);
     * \endcode
     */
    static void computeInitialVelocityUniform();

  private:

};// end of SolverCudaKernels
//----------------------------------------------------------------------------------------------------------------------

#endif /* SOLVER_CUDA_KERNELS_H */

/**
 * @file      KSpaceFirstOrderSolver.h
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file containing the main class of the project responsible for the entire simulation.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 July      2012, 10:27 (created)\n
 *            11 February  2020, 16:14 (revised)
 *
 * @copyright Copyright (C) 2012 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#ifndef KSPACE_FIRST_ORDER_SOLVER_H
#define KSPACE_FIRST_ORDER_SOLVER_H

#include <functional>

#include <Parameters/Parameters.h>

#include <Containers/MatrixContainer.h>
#include <Containers/OutputStreamContainer.h>

#include <MatrixClasses/RealMatrix.h>
#include <MatrixClasses/ComplexMatrix.h>
#include <MatrixClasses/IndexMatrix.h>

#include <Utils/TimeMeasure.h>

#include <KSpaceSolver/SolverCudaKernels.cuh>

/**
 * @class   KSpaceFirstOrderSolver
 * @brief   Class responsible for solving the ultrasound propagation in fluid medium.
 * @details The simulation is based on the k-space first order method. The solver supports 2D and 3D normal medium.
 *           The axisymmetric medium is not supported. This class maintain the whole k-wave (implements the time loop).
 */
class KSpaceFirstOrderSolver
{
  public:
    /// Constructor.
    KSpaceFirstOrderSolver();
    /// Copy constructor not allowed for public.
    KSpaceFirstOrderSolver(const KSpaceFirstOrderSolver&) = delete;
    /// Destructor.
    virtual ~KSpaceFirstOrderSolver();
    /// Operator = not allowed for public.
    KSpaceFirstOrderSolver& operator=(const KSpaceFirstOrderSolver&) = delete;

    /// Memory allocation.
    virtual void allocateMemory();
    /// Memory deallocation.
    virtual void freeMemory();

    /**
     * @brief   Load simulation data.
     * @details If checkpointing is enabled, this may include reading data from checkpoint and output files.
     */
    virtual void loadInputData();

    /**
     * @brief   This method computes the simulation.
     * @details It first initializes FFTs, then runs the pre-processing phase, continues through the simulation time
     *          loop and applies post-process on the data.
     */
    virtual void compute();

    /**
     * @brief  Get host memory usage in MB.
     * @return Host memory usage in MB.
     */
    size_t getHostMemoryUsage() const;
    /**
     * @brief  Get device memory usage in MB.
     * @return Device memory usage in MB.
     */
    size_t getDeviceMemoryUsage() const;
    /**
     * @brief  Get expected disk space usage in MB.
     * @return Expected disk usage by the output file in MB.
     */
    size_t getFileUsage();

    /**
     * @brief  Get code name - release code version.
     * @return Release code version.
     */
    std::string getCodeName() const;

    /// Print the code name and license.
    void   printFullCodeNameAndLicense() const;

    /**
     * @brief  Get total simulation time.
     * @return Total simulation time in seconds.
     */
    double getTotalTime()          const { return mTotalTime.getElapsedTime(); };
    /**
     * @brief  Get total simulation time accumulated over all legs.
     * @return Total execution time in seconds accumulated over all legs.
     */
    double getCumulatedTotalTime() const { return mTotalTime.getElapsedTimeOverAllLegs(); };

  protected:
    /**
     * @brief Check the output file has the correct format and version.
     * @throw ios::failure - If an error happens.
     */
    void checkOutputFile();
    /**
     * @brief Check the checkpoint file has the correct format and version.
     * @throw ios::failure - If an error happens.
     */
    void checkCheckpointFile();

    /// Read the header of the output file and sets the cumulative elapsed time from the first log.
    void readElapsedTimeFromOutputFile();

    /**
     * @brief  Was the loop interrupted to checkpoint?
     * @return true if the simulation has been interrupted.
     */
    bool isCheckpointInterruption() const;

    /// Initialize cuda FFT plans.
    void initializeCufftPlans();
    /**
     * @brief   Compute pre-processing phase.
     * @details Initialize all indices, pre-compute constants such as c^2, rho0Sgx * dt  and create kappa, derivative
     *          and shift operators, PML, absorbEta, absorbTau, absorbNabla1, absorbNabla2  matrices.
     * @tparam  simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void preProcessing();
    /**
     * @brief  Compute the main time loop of the k-space first order solver.
     *
     * @tparam simulationDimension - Dimensionality of the simulation.
     * @tparam rho0ScalarFlag      - Is density homogeneous?
     * @tparam bOnAScalarFlag      - Is nonlinearity homogenous?
     * @tparam c0ScalarFlag        - Is sound speed homogeneous?
     * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
     */
    template<Parameters::SimulationDimension simulationDimension,
             bool                            rho0ScalarFlag,
             bool                            bOnAScalarFlag,
             bool                            c0ScalarFlag,
             bool                            alphaCoefScalarFlag>
    void computeMainLoop();
    /// Post processing and closing the output streams.
    void postProcessing();

    /**
     * @brief   Store sensor data.
     * @details This routine exploits asynchronous behavior. It first performs IO from the (i-1)th step while
     *          waiting for ith step to come to the point of sampling.
     */
    void storeSensorData();
    /// Write statistics and header into the output file.
    void writeOutputDataInfo();
    /// Write checkpoint data and flush aggregated outputs into the output file.
    void writeCheckpointData();

    /// Print progress statistics.
    void printStatistics();

    //---------------------------------------------- Compute velocity ------------------------------------------------//
    /**
     * @brief   Compute new values of acoustic velocity in all used dimensions (UxSgx, UySgy, UzSgz).
     * @details The pressure gradients p_k on staggered grid are stored in kTemp1RealND, kTemp2RealND,
                kTemp3RealND variables.
     * @tparam  simulationDimension - Dimensionality of the simulation.
     * @tparam  rho0ScalarFlag      - Is density homogeneous?
     *
     * <b>Matlab code:</b> \code
     *  p_k = fftn(p);
     *  ux_sgx = bsxfun(@times, pml_x_sgx, ...
     *       bsxfun(@times, pml_x_sgx, ux_sgx) ...
     *       - dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     *  uy_sgy = bsxfun(@times, pml_y_sgy, ...
     *       bsxfun(@times, pml_y_sgy, uy_sgy) ...
     *       - dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     *  uz_sgz = bsxfun(@times, pml_z_sgz, ...
     *       bsxfun(@times, pml_z_sgz, uz_sgz) ...
     *       - dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) ...
     *       );
     \endcode
     */
    template<Parameters::SimulationDimension simulationDimension,
             bool                            rho0ScalarFlag>
    void computeVelocity();

    /**
     * @brief  Compute shifted velocities for --u_non_staggered flag.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  ux_shifted = real(ifft(bsxfun(\@times, x_shift_neg, fft(ux_sgx, [], 1)), [], 1));
     *  uy_shifted = real(ifft(bsxfun(\@times, y_shift_neg, fft(uy_sgy, [], 2)), [], 2));
     *  uz_shifted = real(ifft(bsxfun(\@times, z_shift_neg, fft(uz_sgz, [], 3)), [], 3));
     * \endcode
     */
    void computeShiftedVelocity();

    //----------------------------------------- Compute velocity gradient --------------------------------------------//
    /**
     * @brief  Compute new values of acoustic velocity gradients.
     * @tparam simulationDimension - Dimensionality of the simulation.
     *
     * <b>Matlab code:</b> \code
     *  duxdx = real(ifftn( bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx)) ));
     *  duydy = real(ifftn( bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy)) ));
     *  duzdz = real(ifftn( bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz)) ));
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension>
    void computeVelocityGradient();

    //---------------------------------------------- Compute pressure ------------------------------------------------//
    /**
     * @brief  Compute acoustic pressure for nonlinear case.
     *
     * @tparam simulationDimension - Dimensionality of the simulation.
     * @tparam rho0ScalarFlag      - Is density homogeneous?
     * @tparam bOnAScalarFlag      - Is nonlinearity homogenous?
     * @tparam c0ScalarFlag        - Is sound speed homogeneous?
     * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
     *
     * <b>Matlab code:</b> \code
     *  case 'lossless'
     *    % calculate p using a nonlinear adiabatic equation of state
     *    p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
     *
     *  case 'absorbing'
     *    % calculate p using a nonlinear absorbing equation of state
     *    p = c0.^2 .* (...
     *        (rhox + rhoy + rhoz) ...
     *        + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) ))...
     *        - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) ))...
     *        + medium.BonA .*(rhox + rhoy + rhoz).^2 ./ (2 .* rho0) ...
     *        );
     *
     *  case 'stokes'
     *    % calculate p using a nonlinear absorbing equation of state
     *    % assuming alpha_power = 2
     *    p = c0.^2 .* (...
     *        (rhox + rhoy + rhoz) ...
     *        + absorb_tau .* rho0 .* (duxdx + duydy + duzdz) ...
     *        + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0) ...
     *        );
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension,
             bool                            rho0ScalarFlag,
             bool                            bOnAScalarFlag,
             bool                            c0ScalarFlag,
             bool                            alphaCoefScalarFlag>
    void computePressureNonlinear();

    /**
     * @brief   Compute acoustic pressure for linear case
     * @details Matlab code refers to the power law as absorbing.
     *
     * @tparam simulationDimension - Dimensionality of the simulation.
     * @tparam rho0ScalarFlag      - Is density homogeneous?
     * @tparam bOnAScalarFlag      - Is nonlinearity homogenous?
     * @tparam c0ScalarFlag        - Is sound speed homogeneous?
     * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
     *
     * <b>Matlab code:</b> \code
     *  case 'lossless'
     *     % calculate p using a linear adiabatic equation of state
     *     p = c0.^2 .* (rhox + rhoy + rhoz);
     *
     *  case 'absorbing'
     *     % calculate p using a linear absorbing equation of state
     *     p = c0.^2 .* ( ...
     *          (rhox + rhoy + rhoz) ...
     *           + absorb_tau .* real(ifftn( absorb_nabla1 .* fftn(rho0 .* (duxdx + duydy + duzdz)) )) ...
     *           - absorb_eta .* real(ifftn( absorb_nabla2 .* fftn(rhox + rhoy + rhoz) )) ...
     *      );
     *
     *  case 'stokes'
     *     % calculate p using a nonlinear absorbing equation of state
     *     % assuming alpha_power = 2
     *     p = c0.^2 .* (...
     *         (rhox + rhoy + rhoz) ...
     *          + absorb_tau .* rho0 .* (duxdx + duydy + duzdz) ...
     *          );
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension,
             bool                            rho0ScalarFlag,
             bool                            bOnAScalarFlag,
             bool                            c0ScalarFlag,
             bool                            alphaCoefScalarFlag>
    void computePressureLinear();

    //------------------------------------------------ Add sources ---------------------------------------------------//
    /**
     * @brief  Add in pressure source.
     * @tparam simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void addPressureSource();
    /// Add in all velocity sources.
    void addVelocitySource();

    /**
     * @brief Scale velocity or pressure source.
     *
     * @param [in] scaledSource - Generated scaled source
     * @param [in] sourceInput  - Source input signal
     * @param [in] sourceIndex  - Source geometry
     * @param [in] manyFlag     - How many time series
     */
    void scaleSource(RealMatrix&        scaledSource,
                     const RealMatrix&  sourceInput,
                     const IndexMatrix& sourceIndex,
                     const size_t       manyFlag);

    /**
     * @brief  Calculate initial pressure source.
     *
     * @tparam simulationDimension - Dimensionality of the simulation.
     * @tparam rho0ScalarFlag      - Is density homogeneous?
     * @tparam bOnAScalarFlag      - Is nonlinearity homogenous? - not used.
     * @tparam c0ScalarFlag        - Is sound speed homogeneous?
     *
     * <b>Matlab code:</b> \code
     *  % add the initial pressure to rho as a mass source
     *  p = source.p0;
     *  rhox = source.p0 ./ (3 .* c.^2);
     *  rhoy = source.p0 ./ (3 .* c.^2);
     *  rhoz = source.p0 ./ (3 .* c.^2);
     *
     *  % compute u(t = t1 + dt/2) based on the assumption u(dt/2) = -u(-dt/2)
     *  % which forces u(t = t1) = 0
     *  ux_sgx = dt .* rho0_sgx_inv .* real(ifftn( bsxfun(@times, ddx_k_shift_pos, kappa .* fftn(p)) )) / 2;
     *  uy_sgy = dt .* rho0_sgy_inv .* real(ifftn( bsxfun(@times, ddy_k_shift_pos, kappa .* fftn(p)) )) / 2;
     *  uz_sgz = dt .* rho0_sgz_inv .* real(ifftn( bsxfun(@times, ddz_k_shift_pos, kappa .* fftn(p)) )) / 2;
     * \endcode
     */
    template<Parameters::SimulationDimension simulationDimension,
             bool                            rho0ScalarFlag,
             bool                            bOnAScalarFlag,
             bool                            c0ScalarFlag>
    void addInitialPressureSource();

    /**
     * @brief  Calculate dt ./ rho0 for nonuniform grids.
     * @tparam simulationDimension - Dimensionality of the simulation.
     */
    template<Parameters::SimulationDimension simulationDimension>
    void generateInitialDenisty();

    //----------------------------------------- Generate constant matrices -------------------------------------------//
    /// Generate kappa matrix for lossless medium.
    void generateKappa();
    /// Generate derivative operators (dd{x,y,z}_k_shift_pos, dd{x,y,z}_k_shift_neg).
    void generateDerivativeOperators();
    /// Generate sourceKappa matrix for additive sources.
    void generateSourceKappa();
    /// Generate kappa matrix, absorbNabla1, absorbNabla2 for absorbing medium.
    void generateKappaAndNablas();
    /// Generate absorbTau, absorbEta for heterogenous medium and power law absorption.
    void generateTauAndEta();
    /// Generate absorbTau for heterogenous medium and stokes absorption.
    void generateTau();
    /// Generate shift variables for non-staggered velocity sampling.
    void generateNonStaggeredShiftVariables();
    /// Generate PML and staggered PML.
    void generatePml();
    /// Generate square of velocity.
    void generateC2();

    //----------------------------------------------- Index routines -------------------------------------------------//
    /**
     * @brief Compute 1D index using 3 spatial coordinates and the size of the matrix.
     * @param [in] z              - z coordinate
     * @param [in] y              - y coordinate
     * @param [in] x              - x coordinate
     * @param [in] dimensionSizes - Size of the matrix.
     * @return
     */
    size_t get1DIndex(const size_t          z,
                      const size_t          y,
                      const size_t          x,
                      const DimensionSizes& dimensionSizes) const;

    //--------------------------------------- Getters for temporary matrices -----------------------------------------//
    /**
     * @brief  Get temporary matrix for 1D fft in x
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftX() const
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftX);
    };
    /**
     * @brief  Get temporary matrix for 1D fft in y.
     * @return Temporary complex 2D/3D matrix.
     */
    CufftComplexMatrix& getTempCufftY() const
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftY);;
    };
    /**
     * @brief  Get temporary matrix for 1D fft in z.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftZ() const
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftZ);
    };
    /**
     * @brief  Get temporary matrix for cufft shift.
     * @return Temporary complex 3D matrix.
     */
    CufftComplexMatrix& getTempCufftShift() const
    {
      return mMatrixContainer.getMatrix<CufftComplexMatrix>(MatrixContainer::MatrixIdx::kTempCufftShift);
    };

    //------------------------------------ Getters for real and index matrices ---------------------------------------//
    /**
     * @brief  Get real matrix from the matrix container.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @return Reference to the real matrix.
     */
    RealMatrix& getRealMatrix(const MatrixContainer::MatrixIdx matrixIdx)
    {
      return mMatrixContainer.getMatrix<RealMatrix>(matrixIdx);
    }
    /**
     * @brief  Get real matrix from the container, const version.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @return Reference to the real matrix.
     */
    const RealMatrix& getRealMatrix(const MatrixContainer::MatrixIdx matrixIdx) const
    {
      return mMatrixContainer.getMatrix<RealMatrix>(matrixIdx);
    }

    /**
     * @brief  Get index matrix from the matrix container.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @return Reference to the index matrix.
     */
    IndexMatrix& getIndexMatrix(const MatrixContainer::MatrixIdx matrixIdx)
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(matrixIdx);
    }
    /**
     * @brief  Get index matrix from the container, const version.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @return Reference to the index matrix.
     */
    const IndexMatrix& getIndexMatrix(const MatrixContainer::MatrixIdx matrixIdx) const
    {
      return mMatrixContainer.getMatrix<IndexMatrix>(matrixIdx);
    }

    //----------------------------- Getters for raw real, complex and index matrix data ------------------------------//
    /**
     * @brief  Get the pointer to raw host data of ComplexMatrix.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    FloatComplex* getComplexData(const MatrixContainer::MatrixIdx matrixIdx,
                                 const bool                       present = true)
    {
      return (present) ? mMatrixContainer.getMatrix<ComplexMatrix>(matrixIdx).getComplexHostData() : nullptr;
    }

    /**
     * @brief  Get the pointer to raw host data of ComplexMatrix, const version.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    const FloatComplex* getComplexData(const MatrixContainer::MatrixIdx matrixIdx,
                                       const bool                       present = true) const
    {
      return (present) ? mMatrixContainer.getMatrix<ComplexMatrix>(matrixIdx).getComplexHostData() : nullptr;
    }

    /**
     * @brief  Get the pointer to raw host data of ComplexMatrix.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    float* getRealData(const MatrixContainer::MatrixIdx matrixIdx,
                       const bool                       present = true)
    {
      return (present) ? mMatrixContainer.getMatrix<RealMatrix>(matrixIdx).getHostData() : nullptr;
    }
    /**
     * @brief  Get the pointer to raw host data of RealMatrix, const version.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    const float* getRealData(const MatrixContainer::MatrixIdx matrixIdx,
                             const bool                       present = true) const
    {
      return (present) ? mMatrixContainer.getMatrix<RealMatrix>(matrixIdx).getHostData() : nullptr;
    }

    /**
     * @brief  Get the pointer to raw host data of IndexMatrix.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    size_t* getIndexData(const MatrixContainer::MatrixIdx matrixIdx,
                         const bool                       present = true)
    {
      return (present) ? mMatrixContainer.getMatrix<IndexMatrix>(matrixIdx).getHostData() : nullptr;
    }

    /**
     * @brief  Get the pointer to raw host data of IndexMatrix, const version.
     * @param  [in] matrixIdx - Matrix id in the container.
     * @param  [in] present   - Is the matrix present in the container?
     * @return Pointer to raw data or nullptr if the matrix is not in the container.
     */
    const size_t* getIndexData(const MatrixContainer::MatrixIdx matrixIdx,
                               const bool                       present = true) const
    {
      return (present) ? mMatrixContainer.getMatrix<IndexMatrix>(matrixIdx).getHostData() : nullptr;
    }

  private:
    /// Pointer to computeMainLoop method.
    using ComputeMainLoopFnc = std::function<void(KSpaceFirstOrderSolver&)>;
    /**
     * @brief Map with possible implementations of the ComputeMainLoop.
     *
     * The key is a tuple composed of
     * [Parameters::SimulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag, alphaCoefScalarFlag]
     */
    using ComputeMainLoopImp = std::map<std::tuple<Parameters::SimulationDimension, bool, bool, bool, bool>,
                                        ComputeMainLoopFnc>;

    /// Map with different implementations of ComputeMainLoop.
    static ComputeMainLoopImp sComputeMainLoopImp;

    /// Matrix container with all the matrix classes.
    MatrixContainer       mMatrixContainer;
    /// Output stream container.
    OutputStreamContainer mOutputStreamContainer;

    /// Global parameters of the simulation.
    Parameters& mParameters;

    /// Percentage of the simulation done.
    size_t      mActPercent;

    /// This variable is true when calculating first time step after restore from checkpoint (to allow asynchronous IO).
    bool        mIsTimestepRightAfterRestore;

    /// Total time of the simulation.
    TimeMeasure mTotalTime;
    /// Pre-processing time of the simulation.
    TimeMeasure mPreProcessingTime;
    /// Data load time of the simulation.
    TimeMeasure mDataLoadTime;
    /// Simulation time of the simulation.
    TimeMeasure mSimulationTime;
    /// Post-processing time of the simulation.
    TimeMeasure mPostProcessingTime;
    /// Iteration time of the simulation.
    TimeMeasure mIterationTime;
};// end of KSpaceFirstOrderSolver
//----------------------------------------------------------------------------------------------------------------------

#endif /* KSPACE_FIRST_ORDER_SOLVER_H */

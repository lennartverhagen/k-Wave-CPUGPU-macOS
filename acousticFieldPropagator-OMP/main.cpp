/**
 * @file main.cpp
 *
 * @brief Entry point of the application, simulation, file I/O and reporting
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

#include <iostream>
#include <string>

#include <omp.h>

#include <Hdf5Io/Hdf5Io.h>
#include <Hdf5Io/Hdf5StringAttribute.h>
#include <Kernels/Kernels.h>
#include <Types/FftMatrix.h>
#include <Types/Parameters.h>
#include <Utils/ArgumentParser.h>
#include <Utils/Memory.h>
#include <Utils/Stopwatch.h>
#include <Utils/Terminal.h>
#include <Utils/TerminalUtils.h>
#ifdef KWAVE_VERSION
#  include <Version/Version.h>
#endif

/**
 * @brief Application entry point
 * @param[in] argc – Argument count
 * @param[in] argv – Argument vector
 * @returns EXIT_SUCCESS if successful, EXIT_FAILURE otherwise
 */
int main(int argc, char** argv)
{
  // turn off HDF5 error messages
  H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

  // create output terminals
  Terminal<60> output(std::cout);
  Terminal<60> error(std::cerr);

  // create stopwatches
  Stopwatch totalTime;
  Stopwatch matrixTime;
  Stopwatch loadTime;
  Stopwatch storeTime;
  Stopwatch fftTime;
  Stopwatch preprocessTime;
  Stopwatch kernelTime;
  Stopwatch postprocessTime;

  // peak memory usage in the most outer scope
  std::size_t memoryUsage;

  try
  {
    auto tp = totalTime.getProbe();

    output.printBoldSeparator();
    output.print1C() << "Acoustic Field Propagator (a part of k-Wave"
#ifdef KWAVE_VERSION
                        " "
                     << kKWaveVersion <<
#endif
        ")";
    output.printSeparator();
    printCodeVersion(output);

    // parse input arguments
    ArgumentParser arguments(argc, argv);
    // check for help flag
    if (arguments.helpWanted())
    {
      printUsage(output, argv[0]);
      return EXIT_SUCCESS;
    }
    // if the number of threads was specified, call omp_set_num_threads
    if (arguments.threadCountSpecified())
    {
      omp_set_num_threads(arguments.getThreadCount());
    }
    int threadCount = omp_get_max_threads();
    output.print2C12<std::left, std::left>() << "Number of threads:" >> threadCount;

    output.printBoldSeparator();
    output.print1C() << "Progress:";
    output.printSeparator();
    output.print1C() << "Opening input file...";
    // open the input file and check attributes
    Hdf5Input inputFile(arguments.getInputFileName());
    if (!inputFile.checkAttributes())
    {
      error.printSeparator();
      error.print1C() << "The input file does not contain the required attributes to perform some checks."
                         " The simulation may fail.";
      error.printSeparator();
    }
    // parse scalar values from the input file
    Parameters params;

    output.print1C() << "Reading input parameters...";
    loadTime.getProbe(), inputFile.readParams(params);

    // output the domain sizes
    // the order is swapped to conform with MATLAB which stores the matrices in column major order
    output.print2C11() << "  Domain size:" >> params.size.z() >> " x " >> params.size.y() >> " x " >> params.size.x() >>
        " points";
    output.print2C11() << "  Extended size:" >> params.extended.z() >> " x " >> params.extended.y() >> " x " >>
        params.extended.x() >> " points";
    output.print1C();

    // load input matrices
    output.print1C() << "Allocating space...";
    FftMatrix m((matrixTime.getProbe(), params.extended), threadCount);

    output.print1C() << "Loading input...";
    if (params.complexInput)
    {
      loadTime.getProbe(), inputFile.readComplexMatrix("source_in", m);
    }
    else
    {
      loadTime.getProbe(), inputFile.readMatrix("amp_in", m);
      if (!params.phaseIsScalar)
      {
        loadTime.getProbe(), inputFile.readMatrix("phase_in", m, true);
      }
    }

    // open up the output file
    Hdf5Output outputFile(arguments.getOutputFileName());
    // write basic attributes and number of threads
    outputFile.writeBasicAttributes();
    Hdf5StringAttribute(outputFile, "number_of_cpu_cores").write(std::to_string(threadCount));
    // copy the description from the input file, if present
    Hdf5StringAttribute description(inputFile, "description");
    if (description.exists())
    {
      Hdf5StringAttribute(outputFile, "description").write(description.read());
    }
    // we can close the input file now
    inputFile.close();

    // if the input was separate, we need to run preprocessing kernel
    if (!params.complexInput)
    {
      output.print1C() << "Preprocessing...";
      preprocessTime.getProbe(), Kernels::preprocess(params, m);
#ifdef DEBUG_DATASET
      output.print1C() << "Storing debug matrices...";
      storeTime.getProbe(), outputFile.writeComplexSubMatrix("source_in", m, params.size);
#endif
    }

    // perform forward FFT
    output.print1C() << "Executing forward FFT...";
    fftTime.getProbe(), m.performForwardFft();

#ifndef DEBUG_DATASET
    // apply per-element kernel in the calculated spectrum
    output.print1C() << "Executing computational kernel...";
    kernelTime.getProbe(), Kernels::advanceWaves(params, m);
#else
    {
      output.print1C() << "Allocating debug matrices...";
      FftMatrix k((matrixTime.getProbe(), params.extended), threadCount);
      FftMatrix propagator((matrixTime.getProbe(), params.extended), threadCount);

      output.print1C() << "Executing computational kernel...";
      kernelTime.getProbe(), Kernels::advanceWaves(params, m, k, propagator);

      output.print1C() << "Storing debug matrices...";
      storeTime.getProbe(), outputFile.writeMatrix("k", k);
      storeTime.getProbe(), outputFile.writeComplexMatrix("propagator", propagator);
    }
#endif

    // perform backward FFT; ending up with
    output.print1C() << "Executing backward FFT...";
    fftTime.getProbe(), m.performBackwardFft();

    if (arguments.complexOuptut())
    {
      output.print1C() << "Normalizing the result...";
      postprocessTime.getProbe(), Kernels::normalize(params, m);
      output.print1C() << "Storing the result...";
      storeTime.getProbe(), outputFile.writeComplexSubMatrix("pressure_out", m, params.size);
    }
    else
    {
      // we need some postprocessing to get amplitude and phase separately
#ifdef DEBUG_DATASET
      output.print1C() << "Storing debug matrices...";
      storeTime.getProbe(), outputFile.writeComplexSubMatrix("pressure_out_not_normalized", m, params.size);
#endif
      output.print1C() << "Recovering amplitude and phase...";
      postprocessTime.getProbe(), Kernels::recovery(params, m);

      output.print1C() << "Storing the result...";
      storeTime.getProbe(), outputFile.writeSubMatrix("amp_out", m, params.size);
      storeTime.getProbe(), outputFile.writeSubMatrix("phase_out", m, params.size, true);
    }

    // force the output file flush, stop of the total timer probe, read memory usage
    // and write everything info into the file
    outputFile.flush();
    tp.stop();

    outputFile.writeTimeAttribute("data_loading_phase_execution_time", matrixTime.getTime() + loadTime.getTime());
    outputFile.writeTimeAttribute("pre-processing_phase_execution_time", preprocessTime.getTime());
    outputFile.writeTimeAttribute("simulation_phase_execution_time", fftTime.getTime() + kernelTime.getTime());
    outputFile.writeTimeAttribute("post-processing_phase_execution_time", postprocessTime.getTime());
    outputFile.writeTimeAttribute("data_storing_phase_execution_time", storeTime.getTime());
    outputFile.writeTimeAttribute("total_execution_time", totalTime.getTime());

    memoryUsage = getMaximumMemoryUsageMiB();

    outputFile.writeMemoryAttribute("total_memory_in_use", memoryUsage);
    outputFile.writeMemoryAttribute("peak_core_memory_in_use", static_cast<double>(memoryUsage) / threadCount);
  }
  catch (std::exception& e)
  {
    // print out what went wrong
    printException(error, e);
    // return with error code 1
    return EXIT_FAILURE;
  }

  try
  {
    // print the execution times
    output.printBoldSeparator();
    output.print1C() << "Execution times:";
    output.printSeparator();
    printTime(output, "Matrix initialization", matrixTime);
    printTime(output, "Data load", loadTime);
    printTime(output, "FFT execution", fftTime);
    if (preprocessTime.getTime().count())
    {
      printTime(output, "Preprocessing", preprocessTime);
    }
    printTime(output, "Pressure field calculation", kernelTime);
    printTime(output, "Postprocessing", postprocessTime);
    printTime(output, "Data store", storeTime);
    output.print1C();
    printTime(output, "Total execution", totalTime);
    // print the peak memory usage
    output.printSeparator();
    output.print2C21() << "Peak memory usage:" >> memoryUsage >> " MiB";
    output.printBoldSeparator();
  }
  catch (std::exception& e)
  {
    // print out what went wrong
    printException(error, e);
    // return with error code 1
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

/**
 * @file Utils/ArgumentParser.cpp
 *
 * @brief Command-line argument parser
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
 * Created: 2017-06-13 06:57\n
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

#include <Utils/ArgumentParser.h>

#include <string>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// clang-format off
const std::vector<ArgumentParser::Argument> ArgumentParser::kMandatory{
    {{"-i"}, "-i <input_file_name>", "HDF5 input file", &ArgumentParser::setInputFileName},
    {{"-o"}, "-o <output_file_name>", "HDF5 output file", &ArgumentParser::setOutputFileName}};

const std::vector<ArgumentParser::Argument> ArgumentParser::kOptional{
    {{"-t"}, "-t <num_threads>", "Number of CPU threads", &ArgumentParser::setThreadCount},
    {{"-c", "--complex-pressure"}, "-c", "Output the pressure in a single complex matrix",
        &ArgumentParser::setComplexOuptut},
    {{"-h", "--help"}, "-h", "Display this usage note", &ArgumentParser::setHelpWanted}};
// clang-format on

std::unordered_map<std::string, std::reference_wrapper<const ArgumentParser::Argument>> ArgumentParser::kArgumentMap{};
const bool ArgumentParser::kOptionsPopulated = ArgumentParser::populateOptions();

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Constructor
 */
ArgumentParser::ArgumentParser(int    argc,
                               char** argv)
    : mInputFile(nullptr), mOutputFile(nullptr), mThreadCountValid(false), mComplexOutput(false), mHelpWanted(false)
{
  int index = 1;
  try
  {
    // run callbacks for every argument encountered (callback can move the argv index)
    while (index < argc)
    {
      (this->*(kArgumentMap.at(argv[index]).get().getCallback()))(argc, argv, index);
    }
  }
  catch (std::out_of_range&)
  {
    throw std::runtime_error(std::string("Unrecognized option '") + argv[index] +
                             "', run with '-h' or '--help' to get usage information");
  }

  // final input validation
  if (mHelpWanted)
  {
    // asking for help means no other arguments are mandatory
    return;
  }
  if (!mInputFile && !mOutputFile)
  {
    throw std::runtime_error("Input and output file names are required arguments, run with '-h' or '--help' to get"
                             " usage information");
  }
  if (!mInputFile)
  {
    throw std::runtime_error(
        "Input file name is a required argument, run with '-h' or '--help' to get usage information");
  }
  if (!mOutputFile)
  {
    throw std::runtime_error(
        "Output file name is a required argument, run with '-h' or '--help' to get usage information");
  }
}// end of ArgumentParser::ArgumentParser
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Static method to initialize argument map
 */
bool ArgumentParser::populateOptions()
{
  for (const auto& argument : kMandatory)
  {
    for (const auto& item : argument)
    {
      auto result = kArgumentMap.insert(std::make_pair(item, std::cref(argument)));
      if (!result.second)
      {
        throw std::runtime_error("Conflicting argument '" + item + "'");
      }
    }
  }
  for (const auto& argument : kOptional)
  {
    for (const auto& item : argument)
    {
      auto result = kArgumentMap.insert(std::make_pair(item, std::cref(argument)));
      if (!result.second)
      {
        throw std::runtime_error("Conflicting argument '" + item + "'");
      }
    }
  }
  return true;
}// end of ArgumentParser::populateOptions
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Callback to set input file name
 */
void ArgumentParser::setInputFileName(int    argc,
                                      char** argv,
                                      int&   current)
{
  if (mInputFile)
  {
    throw std::runtime_error("Multiple arguments specifying input file name encountered");
  }

  // check if still within argv and the file name is following
  if (current + 1 >= argc || kArgumentMap.find(argv[current + 1]) != kArgumentMap.end())
  {
    throw std::runtime_error(std::string("Input file name is missing for the '") + argv[current] +
                             "' argument, run with '-h' or '--help' to get usage information");
  }
  // assign
  mInputFile = argv[current + 1];
  // increment the index
  current += 2;
}// end of ArgumentParser::setInputFileName
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Callback to set output file name
 */
void ArgumentParser::setOutputFileName(int    argc,
                                       char** argv,
                                       int&   current)
{
  if (mOutputFile)
  {
    throw std::runtime_error("Multiple arguments specifying output file name encountered");
  }

  // check if still within argv and the file name is following
  if (current + 1 >= argc || kArgumentMap.find(argv[current + 1]) != kArgumentMap.end())
  {
    throw std::runtime_error(std::string("Output file name is missing for the '") + argv[current] +
                             "' argument, run with '-h' or '--help' to get usage information");
  }
  // assign
  mOutputFile = argv[current + 1];
  // increment the index
  current += 2;
}// end of ArgumentParser::setOutputFileName
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Callback to set number of threads
 */
void ArgumentParser::setThreadCount(int    argc,
                                    char** argv,
                                    int&   current)
{
  if (mThreadCountValid)
  {
    throw std::runtime_error("Multiple arguments specifying number of threads encountered");
  }

  // check if still within argv and the file name is following
  if (current + 1 >= argc || kArgumentMap.find(argv[current + 1]) != kArgumentMap.end())
  {
    throw std::runtime_error(std::string("Thread count is missing for the '") + argv[current] +
                             "' argument, run with '-h' or '--help' to get usage information");
  }
  // assign
  try
  {
    std::size_t pos;
    unsigned numThreads = std::stoul(argv[current + 1], &pos);
    // check if the whole input was parsed
    if (argv[current + 1][pos] != '\0')
    {
      throw std::invalid_argument("");
    }
    // conversion unsigned -> int
    mThreadCount = numThreads;
    // check for range (converted number must be still positive)
    if (mThreadCount <= 0)
    {
      throw std::out_of_range("");
    }
    mThreadCountValid = true;
  }
  catch (std::invalid_argument&)
  {
    throw std::runtime_error("Number of threads must be a valid unsigned number, run with '-h' or '--help' to get "
                             "usage information");
  }
  catch (std::out_of_range&)
  {
    throw std::runtime_error("Specified number of threads is too big");
  }

  // increment the index
  current += 2;
}// end of ArgumentParser::setThreadCount
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Callback to set complex output flag
 */
void ArgumentParser::setComplexOuptut(int    argc,
                                      char** argv,
                                      int&   current)
{
  // no check necessary, no parameter following
  mComplexOutput = true;
  // increment the index
  ++current;
}// end of ArgumentParser::setComplexOuptut
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Callback to set help (usage) flag
 */
void ArgumentParser::setHelpWanted(int    argc,
                                   char** argv,
                                   int&   current)
{
  // no check necessary, no parameter following
  mHelpWanted = true;
  // increment the index
  ++current;
}// end of ArgumentParser::setHelpWanted
//----------------------------------------------------------------------------------------------------------------------

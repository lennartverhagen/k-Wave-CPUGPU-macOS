/**
 * @file Utils/TerminalUtils.h
 *
 * @brief Several routines for commonly printed output to Terminal
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
 * Created: 2017-03-22 18:23\n
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

#ifndef TERMINAL_UTILS_H
#define TERMINAL_UTILS_H

#include <exception>

#include <Utils/ArgumentParser.h>
#include <Utils/Stopwatch.h>
#include <Utils/Terminal.h>
#include <Version/Version.h>

/**
 * @brief Function to print Stopwatch accumulated time into terminal
 *
 * Template parameters are automatically deduced.
 *
 * @tparam    kWidth    – Terminal width
 * @tparam    kBufSize  – Terminal buffer size
 * @param[in] output    – Terminal to use
 * @param[in] name      – Name for the time measured by stopwatch
 * @param[in] stopwatch – Stopwatch to print time from
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
void printTime(Terminal<kWidth, kBufSize>& output,
               const char*                 name,
               const Stopwatch&            stopwatch)
{
  std::chrono::duration<double, std::milli> time(stopwatch.getTime());
  output.print2C21() << name << " time:" >> std::fixed >> std::setprecision(2) >> time.count() >> " ms";
}// end of printTime
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Function to print exception message
 *
 * Template parameters are automatically deduced.
 *
 * @tparam    kWidth    – Terminal width
 * @tparam    kBufSize  – Terminal buffer size
 * @param[in] output    – Terminal to use
 * @param[in] exception – Exception to print information about
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
void printException(Terminal<kWidth, kBufSize>& output,
                    const std::exception&       exception)
{
  output.printBoldSeparator();
  output.print1C() << "Following exception occurred during the execution:";
  output.print1C() << "  " << exception.what();
  output.printBoldSeparator();
}// end of printException
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Function to print code version into terminal
 *
 * Template parameters are automatically deduced.
 *
 * @tparam kWidth Terminal width
 * @tparam kBufSize Terminal buffer size
 */
template<std::size_t kWidth,
         std::size_t kBufSize>
void printCodeVersion(Terminal<kWidth, kBufSize>& output)
{
  output.template print2C12<std::left, std::left>() << "Version:" >> kRevisionString;
  output.template print2C12<std::left, std::left>() << "Repository status:" >> kRepositoryStatus;
  output.template print2C12<std::left, std::left>() << "Compiler:" >> kCompilerIdentifier;
  output.template print2C12<std::left, std::left>() << "Built on:" >> kBuildDateTime;
}// end of printCodeVersion
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Function to print application usage
 *
 * Template parameters are automatically deduced.
 *
 * @tparam    kWidth      – Terminal width
 * @tparam    kBufSize    – Terminal buffer size
 * @param[in] output      – Terminal to use
 * @param[in] programName – Name of the program
 */
template<std::size_t kWidth, std::size_t kBufSize>
void printUsage(Terminal<kWidth, kBufSize>& output, const char* const programName)
{
  output.printBoldSeparator();
  {
    auto proxy = output.print1C();
    proxy << "Usage: " << programName;
    for (auto& argument : ArgumentParser::kMandatory)
    {
      proxy << ' ' << argument.getUsage();
    }
    for (auto& argument : ArgumentParser::kOptional)
    {
      proxy << " [" << argument.getUsage() << ']';
    }
  }
  output.print1C();
  output.print1C() << "Mandatory arguments:";
  output.printSeparator();
  for (auto& argument : ArgumentParser::kMandatory)
  {
    output.template print2C11<std::left, std::left>() << "    " << argument.getUsage() >> argument.getDescription();
  }
  output.print1C();
  output.print1C() << "Optional arguments:";
  output.printSeparator();
  for (auto& argument : ArgumentParser::kOptional)
  {
    output.template print2C11<std::left, std::left>() << "    " << argument.getUsage() >> argument.getDescription();
  }
}// end of printUsage
//----------------------------------------------------------------------------------------------------------------------

#endif /* TERMINAL_UTILS_H */

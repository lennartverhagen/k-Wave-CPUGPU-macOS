/**
 * @file Utils/ArgumentParser.h
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

#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <array>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

/**
 * @brief Class to parse program arguments
 *
 * A simple class that uses map and callbacks to parse and check command line arguments for this specific application.
 * Arguments are parsed during the construction and the results can be accessed by the getters.
 */
class ArgumentParser
{
    /**
     * @brief Class representing a single argument
     *
     * Simple class summarizing an argument. It groups possible formats of the argument (i.e. `-h` and `--help`),
     * a usage string (listed in usage summary) and a description (printed in usage too).
     */
    class Argument
    {
      public:
        /// Method pointer type for callbacks
        using SetterCallback = void (ArgumentParser::*)(int argc, char** argv, int& current);

        using const_iterator = std::vector<std::string>::const_iterator;

        /**
         * @brief Argument constructor
         *
         * @param[in] options     – Vector containing all the possible formats the argument can be specified with
         * @param[in] usage       – Argument usage string
         * @param[in] description – Argument description
         * @param[in] callback    – ArgumentParser method to be called when the argument is encountered
         * @throws std::runtime error if there was a problem with parsing
         */
        Argument(const std::vector<std::string>& options,
                 const std::string&              usage,
                 const std::string&              description,
                 SetterCallback callback)
            : mOptions(options),
              mUsage(usage),
              mDescription(description),
              mCallback(callback)
        {}

        /// Description string getter
        const std::string& getDescription() const { return mDescription; }
        /// Usage string getter
        const std::string& getUsage()       const { return mUsage; }
        /// Callback getter
        SetterCallback getCallback()        const { return mCallback; }

        /**
         * @brief Method returning options iterator
         * @returns Iterator to the beginning of the option strings
         */
        const_iterator begin() const { return mOptions.begin(); }

        /**
         * @brief Method returning options iterator
         * @returns Iterator to the beginning of the option strings
         */
        const_iterator cbegin() const { return mOptions.cbegin(); }

        /**
         * @brief Method returning options iterator
         * @returns Iterator to the end of the option strings
         */
        const_iterator end()    const { return mOptions.end(); }

        /**
         * @brief Method returning options iterator
         * @returns Iterator to the end of the option strings
         */
        const_iterator cend()   const { return mOptions.cend(); }

      private:
        /// All the possible options (formats) for the argument
        const std::vector<std::string> mOptions;
        /// Usage string (e.g. `-i <input_file>`)
        const std::string    mUsage;
        /// Description string (e.g. `Input file`)
        const std::string    mDescription;
        /// ArgumentParser::* method callback
        const SetterCallback mCallback;
    };

  public:
    /**
     * @brief Constructor
     *
     * Constructs ArgumentParser object while parsing the command line arguments
     *
     * @param[in] argc – Number of arguments in the vector
     * @param[in] argv – Argument vector
     * @throws    std::runtime_error if parsing fails
     */
    ArgumentParser(int    argc,
                   char** argv);

    /// Destructor
    ~ArgumentParser() {}

    /**
     * @brief Getter method to obtain input file name
     * @returns Pointer to a C-like string in argument vector that is the input file name
     */
    const char* getInputFileName()  const { return mInputFile; }

    /**
     * @brief Getter method to obtain output file name
     * @returns Pointer to a C-like string in argument vector that is the output file name
     */
    const char* getOutputFileName() const { return mOutputFile; }

    /**
     * @brief Method to check whether the thread count was specified
     * @returns True, if the thread count was specified in the arguments, false otherwise
     */
    bool threadCountSpecified()     const { return mThreadCountValid; }

    /**
     * @brief Getter method to obtain number of threads
     *
     * Please check that the value was parsed using threadCountSpecified() before calling this method.
     *
     * @returns Numbers of threads to use
     * @throws  std::runtime_error if the number was not specified
     */
    int getThreadCount() const
    {
      if (!mThreadCountValid)
      {
        throw std::runtime_error("Number of threads was not specified");
      }
      return mThreadCount;
    }

    /**
     * @brief Check whether complex output was requested
     * @returns True, if the complex flag was encountered at least once
     */
    bool complexOuptut() const { return mComplexOutput; }

    /**
     * @brief Check whether help was requested
     * @returns True, if at least one of the help arguments was encountered
     */
    bool helpWanted()    const { return mHelpWanted; }

    /// Static vector of accepted mandatory arguments
    const static std::vector<Argument> kMandatory;
    /// Static vector of accepted optional arguments
    const static std::vector<Argument> kOptional;

  private:
    /**
     * @brief Static method to initialize argument map
     *
     * This method initializes kOptionsPopulated static member with 'true' value as a mean of being called in the
     * initialization phase, before the main() is executed.
     *
     * It takes all the mandatory and optional arguments stored in kMandatory and kOptional static members and
     * fills up the kArgumentMap that is used to lookup known arguments.
     *
     * @returns true
     */
    static bool populateOptions();

    /// Callback to set input file name
    void setInputFileName(int argc, char** argv, int& current);
    /// Callback to set output file name
    void setOutputFileName(int argc, char** argv, int& current);
    /// Callback to set number of threads
    void setThreadCount(int argc, char** argv, int& current);
    /// Callback to set complex output flag
    void setComplexOuptut(int argc, char** argv, int& current);
    /// Callback to set help (usage) flag
    void setHelpWanted(int argc, char** argv, int& current);

    /// Pointer to the input file name
    const char* mInputFile;
    /// Pointer to the output file name
    const char* mOutputFile;
    /// Flag that thread count was parsed
    bool mThreadCountValid;
    /// Parsed thread count (if encountered)
    int  mThreadCount;
    /// Flag indicating the output to be complex
    bool mComplexOutput;
    /// Flag indicating help flag
    bool mHelpWanted;

    /// Map initialized by populateOptions(), containing references to argument objects in kMandatory and kOptional
    /// arrays
    static std::unordered_map<std::string, std::reference_wrapper<const Argument>> kArgumentMap;
    /// Static variable that is initialized last; its initialization invokes populateOptions() and fills up
    /// kArgumentMap
    const static bool kOptionsPopulated;
};// end of ArgumentParser
//----------------------------------------------------------------------------------------------------------------------

#endif /* ARGUMENT_PARSER_H */

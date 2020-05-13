/**
 * @file Utils/Stopwatch.h
 *
 * @brief Accumulating stopwatch with RAII probes
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

#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>
#include <exception>
#include <stdexcept>

/**
 * @brief Class providing a simple scoped stopwatch
 *
 * Object of this file provides an accumulator for measuring time spent in specified parts of the code.
 * A probe needs to be inserted to every part that should be included in the resulting time. After the
 * measurement is done, the accumulated time can be queried using getTime() method.
 */
class Stopwatch
{
  public:
    /// Type used for the internal clock status
    using steady_clock = std::chrono::steady_clock;

    /**
     * @brief Proxy object used for scoped time measurement
     *
     * Object of the class Stopwatch::Probe can only be created by calling Stopwatch::getProbe(). Every
     * object of this class must have an associated Stopwatch object. It captures time upon construction
     * and destruction, accumulating the time spent in a scope into the associated Stopwatch.
     */
    class Probe
    {
        /// friend class relationship required to access Probe constructor from Stopwatch
        friend class Stopwatch;

      public:
        /**
         * @brief Stopwatch::Probe destructor
         *
         * Captures time and accumulates the result into the associated Stopwatch. If the cause of the
         * destruction is an exception, the time will not be modified as the value is considered to be
         * meaningless.
         */
        ~Probe()
        {
          // to handle the situation the probe was moved
          if (!mParent)
          {
            return;
          }
          auto time = steady_clock::now() - mStart;
          if (std::uncaught_exception())
          {
            return;
          }
          mParent->mAccumulated += time;
        }

        /// Copy constructor not allowed
        Probe(const Probe&) = delete;
        /// Operator = not allowed
        Probe& operator=(const Probe&) = delete;

        /// Move constructor
        Probe(Probe&& orig) : mParent(orig.mParent), mStart(orig.mStart)
        {
          // remove the parent reference from the original clock to prevent double measurement
          orig.mParent = nullptr;
        }

        /**
         * @brief Writes a time update to the associated stopwatch
         *
         * Probe continues with the measurement and the associated stopwatch will be updated upon calling
         * `flush()`, `stop()` or during probe destruction.
         *
         * @throws std::runtime_error if called on an inactive probe
         */
        void flush()
        {
          // the following should not occur unless the probe was stopped or using a moved probe object
          if (!mParent)
          {
            throw std::runtime_error("The probe is inactive");
          }
          auto time = steady_clock::now();
          mParent->mAccumulated += time - mStart;
          mStart = time;
        }

        /**
         * @brief Updates the associated stopwatch and deactivates the probe
         *
         * No further updates from the probe will be propagated. If `flush()` or `stop()` is called again it
         * throws an exception.
         *
         * @throws std::runtime_error if called on an inactive probe
         */
        void stop()
        {
          flush();
          // avoid further time accumulation
          mParent = nullptr;
        }

      private:
        /**
         * @brief Stopwatch::Probe constructor
         *
         * Captures starting time.
         *
         * @param[in] parent Reference to the associated Stopwatch object
         */
        Probe(Stopwatch& parent) : mParent(&parent), mStart(steady_clock::now()) {}

        /// Pointer to an associated Stopwatch object, nullptr if the object was moved
        Stopwatch* mParent;
        /// Starting time
        steady_clock::time_point mStart;
    };// end of Stopwatch::Probe
    //-----------------------------------------------------------------------------------------------------------------

    /**
     * @brief Stopwatch constructor
     *
     * Creates a new stopwatch object.
     */
    Stopwatch() : mAccumulated(0) {}
    /// Copy constructor not allowed
    Stopwatch(const Stopwatch&)            = delete;
    /// Move constructor not allowed
    Stopwatch(Stopwatch&&)                 = delete;
    /// Operator = not allowed
    Stopwatch& operator=(const Stopwatch&) = delete;
    /// Stopwatch destructor
    ~Stopwatch() {}

    /**
     * @brief Method to obtain measuring probe
     *
     * This method creates a proxy object meant for scoped time measurement. Its usage is straight-forward,
     * make sure the object's lifetime matches the interval you want to measure.
     *
     * For example, if you want to measure a function time, you should put the probe in the scope of the
     * function, at the beginning. For statements, you can use comma operator to measure execution time.
     *
     * The probe object *must not* outlive the associated Stopwatch object it was created from. That would
     * lead to an unspecified behaviour.
     *
     * @returns Proxy object for scoped time measurement
     */
    Probe getProbe() { return Probe(*this); }

    /**
     * @brief Method to get accumulated time
     *
     * Method returns duration that was accumulated from the probes so far. This does not include partial
     * intervals except when probes are manually flushed, stopped or properly destructed without pending exceptions.
     *
     * @returns Accumulated duration
     */
    steady_clock::duration getTime() const { return mAccumulated; }

  private:
    /// Accumulated time
    steady_clock::duration mAccumulated;
};// end of Stopwatch
//----------------------------------------------------------------------------------------------------------------------

#endif /* STOPWATCH_H */

/**
 * @file Utils/HostnameLin.cpp
 *
 * @brief Linux-specific implementation of obtaining hostname and FQDN
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
 * Created: 2020-02-17 18:38\n
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

#include <Utils/Hostname.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief Get host name, Linux version
 */
std::string getHostname()
{
  std::vector<char> hostname(256, '\0');
  if (gethostname(hostname.data(), hostname.size() - 1) != 0)
  {
    return "<unknown>";
  }

  // use the obtained hostname and try to obtain an associated FQDN too
  addrinfo hints;
  addrinfo* infoPtr;

  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags  = AI_CANONNAME;

  if (getaddrinfo(hostname.data(), nullptr, &hints, &infoPtr) != 0)
  {
    return hostname.data();
  }
  // ensure resource release
  std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> info(infoPtr, &freeaddrinfo);

  // get the canonical name from the first addrinfo result and return
  std::stringstream result;
  result << hostname.data() << " (" << info->ai_canonname << ")";

  return result.str();
}// end of getHostname
//----------------------------------------------------------------------------------------------------------------------

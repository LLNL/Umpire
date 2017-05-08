#ifndef UMPIRE_MemorySpaceDescriptors_HPP
#define UMPIRE_MemorySpaceDescriptors_HPP

#include "umpire/MemorySpace.hpp"

namespace umpire {

enum class MemorySpaceDescriptors : MemorySpace {
  HOST = HostSpace,
  DEVICE = DeviceSpace
};

} // end of namespace umpire

#endif // UMPIRE_MemorySpaceDescriptors_HPP

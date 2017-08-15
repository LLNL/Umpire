#ifndef UMPIRE_DeviceSpaceFactory_HPP
#define UMPIRE_DeviceSpaceFactory_HPP

#include "umpire/AllocatorFactory.hpp"

namespace umpire {
namespace space {


class DeviceSpaceFactory :
  public AllocatorFactory
{
  bool isValidAllocatorFor(const std::string& name);
  std::shared_ptr<Allocator> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_DeviceSpaceFactory_HPP

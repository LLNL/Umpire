#ifndef UMPIRE_DeviceResourceFactory_HPP
#define UMPIRE_DeviceResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {


class DeviceResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name);
  std::shared_ptr<MemoryResource> create();
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_DeviceResourceFactory_HPP

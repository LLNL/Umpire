#ifndef UMPIRE_HostResourceFactory_HPP
#define UMPIRE_HostResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {


class HostResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name);
  std::shared_ptr<MemoryResource> create();
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_HostResourceFactory_HPP

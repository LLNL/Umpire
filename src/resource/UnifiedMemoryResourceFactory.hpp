#ifndef UMPIRE_UnifiedMemoryResourceFactory_HPP
#define UMPIRE_UnifiedMemoryResourceFactory_HPP

#include "umpire/strategy/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {

class UnifiedMemoryResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name);
  std::shared_ptr<MemoryResource> create();
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_UnifiedMemoryResourceFactory_HPP

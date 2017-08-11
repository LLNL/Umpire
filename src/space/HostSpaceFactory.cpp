#include "umpire/space/MemorySpaceRegistry.hpp"
#include "umpire/space/HostSpaceFactory.hpp"
#include "umpire/space/MemorySpace.hpp"

#include "umpire/alloc/MallocAllocator.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

namespace umpire {
namespace space {

HostSpaceFactory::HostSpaceFactory() {
}

void 
HostSpaceFactory::registerFactory(MemorySpaceRegistry& registry) {
  UMPIRE_LOG("Registering HostSpaceFactory");
  registry.registerMemorySpaceFactory("HOST", std::make_shared<HostSpaceFactory>());
}

std::shared_ptr<MemorySpace> HostSpaceFactory::create() {
  UMPIRE_LOG("Creating MemorySpace");
  return std::make_shared<MemorySpace<alloc::MallocAllocator>("HOST");
}

} // end of namespace space
} // end of namespace umpire

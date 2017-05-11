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
  registry.registerMemorySpaceFactory("HOST", new HostSpaceFactory());
}

MemorySpace* HostSpaceFactory::create() {
  UMPIRE_LOG("Creating MemorySpace");
  MemorySpace* space = new MemorySpace(new alloc::MallocAllocator());

  return space;
}

} // end of namespace space
} // end of namespace umpire

#include "umpire/space/MemorySpaceRegistry.hpp"
#include "umpire/space/DeviceSpaceFactory.hpp"
#include "umpire/space/MemorySpace.hpp"

#include "umpire/alloc/MallocAllocator.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

namespace umpire {
namespace space {

DeviceSpaceFactory::DeviceSpaceFactory() {
}

void 
DeviceSpaceFactory::registerFactory(MemorySpaceRegistry& registry) {
  UMPIRE_LOG("Registering DeviceSpaceFactory");
  registry.registerMemorySpaceFactory("DEVICE", std::make_shared<DeviceSpaceFactory>());
}

std::shared_ptr<MemorySpace> DeviceSpaceFactory::create() {
  UMPIRE_LOG("Creating MemorySpace");
  return std::make_shared<MemorySpace>(new alloc::CudaMallocAllocator());
}

} // end of namespace space
} // end of namespace umpire

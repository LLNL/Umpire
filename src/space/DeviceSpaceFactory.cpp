#include "umpire/space/DeviceSpaceFactory.hpp"

#include "umpire/space/MemorySpaceRegistry.hpp"
#include "umpire/alloc/CudaMallocAllocator.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

#include <cuda.h>

namespace umpire {
namespace space {

DeviceSpaceFactory::DeviceSpaceFactory() {
}

void 
DeviceSpaceFactory::registerFactory(MemorySpaceRegistry& registry) {
  UMPIRE_LOG("Registering DeviceSpaceFactory");

  int num_cuda_devices;
  cudaGetDeviceCount(&num_cuda_devices);

  if (num_cuda_devices > 0) {
    registry.registerMemorySpaceFactory(
        "DEVICE",
        std::make_shared<DeviceSpaceFactory>());
  } else {
    UMPIRE_LOG("No CUDA devices found, \"DEVICE\" space not available.");
  }
}

std::shared_ptr<MemorySpace> DeviceSpaceFactory::create() {
  UMPIRE_LOG("Creating MemorySpace");
  return std::make_shared<MemorySpace>(
      "DEVICE",
      new alloc::CudaMallocAllocator());
}

} // end of namespace space
} // end of namespace umpire

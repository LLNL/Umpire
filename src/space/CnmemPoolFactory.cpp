#include "umpire/space/CnmemPoolFactory.hpp"

#include "umpire/space/MemorySpaceRegistry.hpp"

#include "umpire/space/CnmemPool.hpp"

#include "umpire/util/Macros.hpp"

#include <iostream>

#include <cuda_runtime_api.h>

namespace umpire {
namespace space {

CnmemPoolFactory::CnmemPoolFactory() {
}

void 
CnmemPoolFactory::registerFactory(MemorySpaceRegistry& registry) {
  UMPIRE_LOG("Registering CnmemPoolFactory");

  int num_cuda_devices;
  cudaGetDeviceCount(&num_cuda_devices);

  if (num_cuda_devices > 0) {
    registry.registerMemorySpaceFactory(
        "CNMEM",
        std::make_shared<CnmemPoolFactory>());
  } else {
    UMPIRE_LOG("No CUDA devices found, \"CNMEM\" space not available.");
  }
}

std::shared_ptr<MemorySpace> CnmemPoolFactory::create() {
  UMPIRE_LOG("Creating MemorySpace");
  return std::make_shared<CnmemPool>("CNMEM");
}

} // end of namespace space
} // end of namespace umpire

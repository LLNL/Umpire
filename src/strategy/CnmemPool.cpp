#include "umpire/strategy/CnmemPool.hpp"

#include "umpire/tpl/cnmem/cnmem.h"

namespace umpire {
namespace strategy {

CnmemPool::CnmemPool(std::shared_ptr<umpire::Allocator>& alloc)
  : AllocationStrategy(alloc)
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  cnmemDevice_t cnmem_device;
  std::memset(&cnmem_device, 0, sizeof(cnmem_device));
  cnmem_device.size = static_cast<size_t>(0.8 * props.totalGlobalMem);
  cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);
}

void* CnmemPool::allocate(size_t bytes)
{
  void* ret;
  cnmemMalloc(&ret, bytes, NULL);
  return ret;
}

void CnmemPool::deallocate(void* ptr)
{
  cnmemFree(ptr, NULL);
}

} // end of namespace space
} // end of namespace umpire

#ifndef UMPIRE_CnmemAllocator_HPP
#define UMPIRE_CnmemAllocator_HPP

#include "umpire/tpl/cnmem/cnmem.h"

namespace umpire {
namespace alloc {

  -CnmemPool::CnmemPool(const std::string& name)
-  : MemorySpace(name, nullptr)
-{
-  cudaDeviceProp props;
-  cudaGetDeviceProperties(&props, 0);
-  cnmemDevice_t cnmem_device;
-  std::memset(&cnmem_device, 0, sizeof(cnmem_device));
-  cnmem_device.size = static_cast<size_t>(0.8 * props.totalGlobalMem);
-  cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);
-}
-
-void* CnmemPool::allocate(size_t bytes)
-{
-  void* ret;
-  cnmemMalloc(&ret, bytes, NULL);
-  return ret;
-}
-
-void CnmemPool::free(void* ptr)
-{
-  cnmemFree(ptr, NULL);

struct CnmemAllocator :
{
  void* allocate(size_t bytes)
  {
    void* ptr;
    ::cnmemMalloc(&ptr, bytes, NULL);
    return ret;
  }

  void* deallocate(void* ptr)
  {
    cnmemFree(ptr, NULL);
  }
}

} // end of namespace alloc
} // end of namespace umpire

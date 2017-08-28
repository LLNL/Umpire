#ifndef UMPIRE_CnmemAllocator_HPP
#define UMPIRE_CnmemAllocator_HPP

#include "umpire/tpl/cnmem/cnmem.h"

#include <cstring>

namespace umpire {
namespace alloc {

struct CnmemAllocator
{
  void* allocate(size_t bytes)
  {
    static bool initialized = false;

    if (!initialized) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, 0);
      cnmemDevice_t cnmem_device;
      std::memset(&cnmem_device, 0, sizeof(cnmem_device));
      cnmem_device.size = static_cast<size_t>(0.8 * props.totalGlobalMem);
      cnmemInit(1, &cnmem_device, CNMEM_FLAGS_DEFAULT);

      initialized = true;
    }

    void* ptr;
    ::cnmemMalloc(&ptr, bytes, NULL);
    return ptr;
  }

  void deallocate(void* ptr)
  {
    ::cnmemFree(ptr, NULL);
  }
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_CnmemAllocator_HPP

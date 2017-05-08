#ifndef UMPIRE_GpuSpace_HPP
#define UMPIRE_GpuSpace_HPP

#include "umpire/MemorySpace.hpp"

namespace umpire {

class GpuSpace : public MemorySpace {
  public:
    GpuSpace();
    void* allocate(size_t bytes);
    void free(void* ptr);

    void getTotalSize();
    
    void getProperties();
    void getRemainingSize();
    std::string getDescriptor();
    
    void setDefaultAllocator(MemoryAllocator allocator);
    MemoryAllocator getDefaultAllocator();
    
    std::vector<MemoryAllocator> getAllocators();
};

} // end of namespace umpire

#endif // UMPIRE_GpuSpace_HPP

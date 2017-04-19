#ifndef UMPIRE_HostSpace_HPP
#define UMPIRE_HostSpace_HPP

#include "umpire/MemorySpace.hpp"

namespace umpire {

class HostSpace : public MemorySpace {
  public:
    HostSpace();
    void* allocate(size_t bytes);
    void free(void* ptr);

    void getTotalSize();
    
    void getProperties();
    void getRemainingSize();
    std::string getDescriptor();
    
    void setDefaultAllocator(MemoryAllocator& allocator);
    MemoryAllocator& getDefaultAllocator();
    
    std::vector<MemoryAllocator*> getAllocators();
};

} // end of namespace umpire

#endif // UMPIRE_HostSpace_HPP

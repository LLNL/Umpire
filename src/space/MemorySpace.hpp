#ifndef UMPIRE_MemorySpace_HPP
#define UMPIRE_MemorySpace_HPP

#include "umpire/MemoryAllocator.hpp"
#include "umpire/AllocationRecord.hpp"

#include <vector>
#include <map>
#include <string>

namespace umpire {

class MemorySpace {
  public: 
    virtual void* allocate(size_t bytes) = 0;
    virtual void free(void* ptr) = 0;

    virtual void getTotalSize() = 0;
    
    virtual void getProperties() = 0;
    virtual void getRemainingSize() = 0;
    //virtual std::string getDescriptor();
    
    virtual void setDefaultAllocator(MemoryAllocator& allocator) = 0;
    virtual MemoryAllocator& getDefaultAllocator() = 0;
    
    virtual std::vector<MemoryAllocator*> getAllocators() = 0;

  protected: 
    std::string m_descriptor;
    std::map<void*, AllocationRecord> m_allocations;
    std::vector<MemoryAllocator*> m_allocators;
    MemoryAllocator* m_default_allocator;
};

} // end of namespace umpire

#endif // UMPIRE_MemorySpace_HPP

#ifndef UMPIRE_MemorySpace_HPP
#define UMPIRE_MemorySpace_HPP

#include "umpire/Allocator.hpp"
#include "umpire/AllocationRecord.hpp"
#include "umpire/alloc/MemoryAllocator.hpp"

#include <vector>
#include <map>
#include <string>

namespace umpire {
namespace space {

class MemorySpace : public Allocator {
  public: 
    MemorySpace();

    MemorySpace(alloc::MemoryAllocator* allocator);

    virtual void* allocate(size_t bytes);
    virtual void free(void* ptr);

    //virtual void* allocate(size_t bytes, Allocator* allocator);

    virtual void getTotalSize();
    
    virtual void getProperties();
    virtual void getRemainingSize();
    virtual std::string getDescriptor();

    //virtual void addAllocator(alloc::MemoryAllocator* allocator);
    
    virtual void setDefaultAllocator(alloc::MemoryAllocator* allocator);
    virtual alloc::MemoryAllocator& getDefaultAllocator();
    
    virtual std::vector<alloc::MemoryAllocator*> getAllocators();

  protected: 
    std::string m_descriptor;
    std::map<void*, AllocationRecord> m_allocations;
    std::vector<alloc::MemoryAllocator*> m_allocators;
    alloc::MemoryAllocator* m_default_allocator;
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_MemorySpace_HPP

#ifndef UMPIRE_MemorySpace_HPP
#define UMPIRE_MemorySpace_HPP

#include "umpire/Allocator.hpp"
#include "umpire/AllocationRecord.hpp"
#include "umpire/alloc/MemoryAllocator.hpp"

#include <vector>
#include <map>
#include <string>
#include <memory>

namespace umpire {

class ResourceManager;

namespace space {

class MemorySpace : 
  public Allocator, 
  public std::enable_shared_from_this<MemorySpace> 
{
  public: 
    MemorySpace(const std::string& name, alloc::MemoryAllocator* allocator);

    virtual void* allocate(size_t bytes);
    virtual void free(void* ptr);

    virtual void getTotalSize();
    
    virtual void getProperties();
    virtual void getRemainingSize();
    virtual std::string getDescriptor();

    virtual void setDefaultAllocator(alloc::MemoryAllocator* allocator);
    virtual alloc::MemoryAllocator& getDefaultAllocator();

  protected: 
    std::string m_descriptor;

    std::map<void*, alloc::MemoryAllocator*> m_allocations;

    alloc::MemoryAllocator* m_default_allocator;
  private:
    MemorySpace();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_MemorySpace_HPP

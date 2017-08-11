#ifndef UMPIRE_MemorySpace_HPP
#define UMPIRE_MemorySpace_HPP

#include "umpire/Allocator.hpp"
#include "umpire/AllocationRecord.hpp"

#include <vector>
#include <map>
#include <string>
#include <memory>

namespace umpire {

class ResourceManager;

namespace space {

template <typename _allocator>
class MemorySpace : 
  public Allocator, 
  public std::enable_shared_from_this<MemorySpace<_allocator> >
{

  public: 
    MemorySpace(const std::string& name);

    virtual void* allocate(size_t bytes);
    virtual void deallocate(void* ptr);

    virtual void getTotalSize();
    
    virtual void getProperties();
    virtual void getRemainingSize();
    virtual std::string getDescriptor();

  protected: 
    MemorySpace();

    std::string m_descriptor;

    _allocator m_allocator;
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_MemorySpace_HPP

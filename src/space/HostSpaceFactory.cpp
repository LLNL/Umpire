#include "umpire/space/HostSpaceFactory.hpp"

#include "umpire/alloc/MallocAllocator.hpp"

namespace umpire {
namespace space {

HostSpaceFactory::isValidAllocatorFor(const std::string& name)
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<Allocator>
HostSpaceFactory::create()
{
  return std::make_shared<umpire::MemorySpace<alloc::MallocAllocator> >();
}

} // end of namespace space
} // end of namespace umpire

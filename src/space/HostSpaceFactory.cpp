#include "umpire/space/HostSpaceFactory.hpp"

#include "umpire/space/MemorySpace.hpp"
#include "umpire/alloc/MallocAllocator.hpp"

namespace umpire {
namespace space {

bool
HostSpaceFactory::isValidAllocatorFor(const std::string& name)
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<AllocatorInterface>
HostSpaceFactory::create()
{
  return std::make_shared<MemorySpace<alloc::MallocAllocator> >();
}

} // end of namespace space
} // end of namespace umpire

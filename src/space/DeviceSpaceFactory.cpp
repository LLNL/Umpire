#include "umpire/space/DeviceSpaceFactory.hpp"

#include "umpire/alloc/CudaMallocAllocator.hpp"

namespace umpire {
namespace space {

DeviceSpaceFactory::isValidAllocatorFor(const std::string& name)
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<Allocator>
DeviceSpaceFactory::create()
{
  return std::make_shared<umpire::MemorySpace<alloc::CudaMallocAllocator> >();
}

} // end of namespace space
} // end of namespace umpire

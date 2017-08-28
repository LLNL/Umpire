#include "umpire/space/DeviceSpaceFactory.hpp"

#include "umpire/alloc/CnmemAllocator.hpp"

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

std::shared_ptr<AllocatorInterface>
DeviceSpaceFactory::create()
{
  return std::make_shared<umpire::MemorySpace<alloc::CnmemAllocator> >();
}

} // end of namespace space
} // end of namespace umpire

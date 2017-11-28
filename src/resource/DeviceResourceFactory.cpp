#include "umpire/resource/DeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/CnmemAllocator.hpp"

namespace umpire {
namespace resource {

bool
DeviceResourceFactory::isValidMemoryResourceFor(const std::string& name)
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
DeviceResourceFactory::create()
{
  return std::make_shared<resource::DefaultMemoryResource<alloc::CnmemAllocator> >(Platform::cuda);
}

} // end of namespace resource
} // end of namespace umpire

#include "umpire/resource/DeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/CudaMallocAllocator.hpp"

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
DeviceResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  return std::make_shared<resource::DefaultMemoryResource<alloc::CudaMallocAllocator> >(Platform::cuda, "DEVICE", id);
}

} // end of namespace resource
} // end of namespace umpire

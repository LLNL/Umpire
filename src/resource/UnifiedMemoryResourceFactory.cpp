#include "umpire/resource/UnifiedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

namespace umpire {
namespace resource {

bool
UnifiedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
{
  if (name.compare("UM") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
UnifiedMemoryResourceFactory::create()
{
  return std::make_shared<resource::DefaultMemoryResource<alloc::CudaMallocManagedAllocator> >(Platform::cuda, "UM");
}

} // end of namespace resource
} // end of namespace umpire

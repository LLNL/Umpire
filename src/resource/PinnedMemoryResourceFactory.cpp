#include "umpire/resource/PinnedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/CudaPinnedAllocator.hpp"

namespace umpire {
namespace resource {

bool
PinnedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
{
  if (name.compare("PINNED") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
PinnedMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  return std::make_shared<resource::DefaultMemoryResource<alloc::CudaPinnedAllocator> >(Platform::cuda, "PINNED", id);
}

} // end of namespace resource
} // end of namespace umpire

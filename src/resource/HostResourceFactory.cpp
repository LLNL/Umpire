#include "umpire/resource/HostResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/MallocAllocator.hpp"

namespace umpire {
namespace resource {

bool
HostResourceFactory::isValidMemoryResourceFor(const std::string& name)
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
HostResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  return std::make_shared<DefaultMemoryResource<alloc::MallocAllocator> >(Platform::cpu, "HOST", id);
}

} // end of namespace resource
} // end of namespace umpire

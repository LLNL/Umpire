#include "umpire/resource/MemoryResource.hpp"

namespace umpire {
namespace resource {

MemoryResource::MemoryResource(const std::string& name, int id) :
  strategy::AllocationStrategy(name, id)
{
}

} // end of namespace resource
} // end of namespace umpire

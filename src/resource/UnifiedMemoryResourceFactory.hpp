#ifndef UMPIRE_UnifiedMemoryResourceFactory_HPP
#define UMPIRE_UnifiedMemoryResourceFactory_HPP

#include "umpire/resource/MemoryResourceFactory.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Factory class to construct a MemoryResource that uses NVIDIA
 * "unified" memory, accesible from both the CPU and NVIDIA GPUs.
 */
class UnifiedMemoryResourceFactory :
  public MemoryResourceFactory
{
  bool isValidMemoryResourceFor(const std::string& name);
  std::shared_ptr<MemoryResource> create(const std::string& name, int id);
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_UnifiedMemoryResourceFactory_HPP

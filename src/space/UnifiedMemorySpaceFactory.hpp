#ifndef UMPIRE_UnifiedMemorySpaceFactory_HPP
#define UMPIRE_UnifiedMemorySpaceFactory_HPP

#include "umpire/AllocatorFactory.hpp"

namespace umpire {
namespace space {


class UnifiedMemorySpaceFactory :
  public AllocatorFactory
{
  bool isValidAllocatorFor(const std::string& name);
  std::shared_ptr<AllocatorInterface> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_UnifiedMemorySpaceFactory_HPP

#ifndef UMPIRE_HostSpaceFactory_HPP
#define UMPIRE_HostSpaceFactory_HPP

#include "umpire/AllocatorFactory.hpp"

namespace umpire {
namespace space {


class HostSpaceFactory :
  public AllocatorFactory
{
  bool isValidAllocatorFor(const std::string& name);
  std::shared_ptr<AllocatorInterface> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_HostSpaceFactory_HPP

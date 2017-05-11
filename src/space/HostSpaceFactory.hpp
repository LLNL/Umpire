#ifndef UMPIRE_HostSpaceFactory_HPP
#define UMPIRE_HostSpaceFactory_HPP

#include "umpire/space/MemorySpaceFactory.hpp"

namespace umpire {
namespace space {

class HostSpaceFactory : public MemorySpaceFactory {
  public:
    HostSpaceFactory();
    void registerFactory(MemorySpaceRegistry& registry);
    MemorySpace* create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_HostSpaceFactory_HPP

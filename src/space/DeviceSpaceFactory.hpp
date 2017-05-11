#ifndef UMPIRE_DeviceSpaceFactory_HPP
#define UMPIRE_DeviceSpaceFactory_HPP

#include "umpire/space/MemorySpaceFactory.hpp"

namespace umpire {
namespace space {

class DeviceSpaceFactory : public MemorySpaceFactory {
  public:
    DeviceSpaceFactory();
    void registerFactory(MemorySpaceRegistry& registry);
    std::shared_ptr<MemorySpace> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_DeviceSpaceFactory_HPP

#ifndef UMPIRE_MemorySpaceFactory_HPP
#define UMPIRE_MemorySpaceFactory_HPP

#include "umpire/space/MemorySpace.hpp"

#include <memory>

namespace umpire {
namespace space {

class MemorySpaceRegistry;

class MemorySpaceFactory {
  public:
    virtual std::shared_ptr<MemorySpace> create() = 0;
    virtual void registerFactory(MemorySpaceRegistry& registry) = 0; 
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_MemorySpaceFactory_HPP

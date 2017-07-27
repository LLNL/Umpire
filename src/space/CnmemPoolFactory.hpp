#ifndef UMPIRE_CnmemPoolFactory_HPP
#define UMPIRE_CnmemPoolFactory_HPP

#include "umpire/space/MemorySpaceFactory.hpp"
#include "umpire/space/MemorySpaceRegistry.hpp"

namespace umpire {
namespace space {

class CnmemPoolFactory : public MemorySpaceFactory {
  public:
    CnmemPoolFactory();
    void registerFactory(MemorySpaceRegistry& registry);
    std::shared_ptr<MemorySpace> create();
};

} // end of namespace space
} // end of namespace umpire

#endif // UMPIRE_CnmemPoolFactory_HPP

#ifndef UMPIRE_AllocatorRegistry_HPP
#define UMPIRE_AllocatorRegistry_HPP

#include "umpire/Allocator.hpp"
#include "umpire/AllocatorFactory.hpp"

#include <memory>
#include <list>

namespace umpire {

class AllocatorRegistry {
  public:
    static AllocatorRegistry& getInstance();

    std::shared_ptr<umpire::Allocator> makeAllocator(const std::string& name);

    void registerAllocator(std::shared_ptr<AllocatorFactory> factory);

  protected:
    AllocatorRegistry();

  private:
    static AllocatorRegistry* s_allocator_registry_instance;

    std::list<std::shared_ptr<AllocatorFactory> > m_allocator_factories;
};

} // end of namespace umpire

#endif // UMPIRE_AllocatorRegistry_HPP

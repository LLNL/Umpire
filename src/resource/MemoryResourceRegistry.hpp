#ifndef UMPIRE_MemoryResourceRegistry_HPP
#define UMPIRE_MemoryResourceRegistry_HPP

#include "umpire/resource/MemoryResource.hpp"
#include "umpire/resource/MemoryResourceFactory.hpp"

#include <memory>
#include <list>

namespace umpire {
namespace resource {

class MemoryResourceRegistry {
  public:
    static MemoryResourceRegistry& getInstance();

    std::shared_ptr<umpire::resource::MemoryResource> makeMemoryResource(const std::string& name, int id);

    void registerMemoryResource(std::shared_ptr<MemoryResourceFactory>&& factory);

  protected:
    MemoryResourceRegistry();

  private:
    static MemoryResourceRegistry* s_allocator_registry_instance;

    std::list<std::shared_ptr<MemoryResourceFactory> > m_allocator_factories;
};

} // end of namespace resource
} // end of namespace umpire

#endif // UMPIRE_MemoryResourceRegistry_HPP

#include "umpire/AllocatorRegistry.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {

AllocatorRegistry* AllocatorRegistry::s_allocator_registry_instance = nullptr;

AllocatorRegistry&
AllocatorRegistry::getInstance()
{
  if (!s_allocator_registry_instance) {
    s_allocator_registry_instance = new AllocatorRegistry();
  }

  return *s_allocator_registry_instance;
}

AllocatorRegistry::AllocatorRegistry() :
  m_allocator_factories()
{
}

void
AllocatorRegistry::registerAllocator(std::shared_ptr<AllocatorFactory> factory)
{
  m_allocator_factories.push_back(factory);
}

std::shared_ptr<umpire::Allocator>
AllocatorRegistry::makeAllocator(const std::string& name)
{
  for (auto allocator_factory : m_allocator_factories) {
    if (allocator_factory->isValidAllocatorFor(name)) {
        return allocator_factory->create();
    }
  }

  UMPIRE_ERROR("Allocator " << name << " not found");
}

} // end of namespace umpire

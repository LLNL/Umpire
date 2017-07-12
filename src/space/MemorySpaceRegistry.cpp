#include "umpire/space/MemorySpaceRegistry.hpp"
#include "umpire/space/HostSpaceFactory.hpp"
#include "umpire/space/DeviceSpaceFactory.hpp"

#include "umpire/util/Macros.hpp"


namespace umpire {
namespace space {

MemorySpaceRegistry* MemorySpaceRegistry::s_memory_space_registry_instance = nullptr;

MemorySpaceRegistry&
MemorySpaceRegistry::getInstance()
{
  if (!s_memory_space_registry_instance) {
    s_memory_space_registry_instance = new MemorySpaceRegistry();
  }

  return *s_memory_space_registry_instance;
}

MemorySpaceRegistry::MemorySpaceRegistry() :
  m_space_factories(),
  m_space_factory_names()
{
  buildRegistry();
}

void MemorySpaceRegistry::buildRegistry()
{
  {
    std::shared_ptr<HostSpaceFactory> f = std::make_shared<HostSpaceFactory>();
    f->registerFactory(*this);
  }

#if defined(HAVE_CUDA)
  {
    std::shared_ptr<DeviceSpaceFactory> f = std::make_shared<DeviceSpaceFactory>();
    f->registerFactory(*this);
  }
#endif

#if defined(HAVE_KNL)
#endif
}

void
MemorySpaceRegistry::registerMemorySpaceFactory(
    const std::string& name, 
    std::shared_ptr<MemorySpaceFactory> factory)
{
  if (m_space_factories.find(name) != m_space_factories.end()) {
    UMPIRE_ERROR("Duplicate MemorySpaceFactories registered with MemorySpaceRegistry");
  }

  m_space_factories[name] = factory;
  m_space_factory_names.push_back(name);
}

std::vector<std::string>
MemorySpaceRegistry::getMemorySpaceFactoryNames()
{
  return m_space_factory_names;
}

std::shared_ptr<MemorySpaceFactory>
MemorySpaceRegistry::getMemorySpaceFactory(const std::string& name)
{
  if (m_space_factories.find(name) != m_space_factories.end()) {
    return m_space_factories[name];
  } else {
    UMPIRE_ERROR("MemorySpaceRegistry: cannot find space with name = " << name);
  }
}


} // end of namespace space
} // end of namespace umpire

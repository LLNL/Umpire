#include "umpire/ResourceManager.hpp"

#include "umpire/space/MemorySpaceRegistry.hpp"

namespace umpire {

ResourceManager* ResourceManager::s_resource_manager_instance = nullptr;

ResourceManager&
ResourceManager::getInstance()
{
  if (!s_resource_manager_instance) {
    s_resource_manager_instance = new ResourceManager();
  }

  return *s_resource_manager_instance;
}

ResourceManager::ResourceManager() :
  m_spaces(),
  m_allocation_spaces()
{
  umpire::space::MemorySpaceRegistry& registry =
    umpire::space::MemorySpaceRegistry::getInstance();

  for (std::string factory_name : registry.getMemorySpaceFactoryNames()) {
    m_spaces[factory_name] = registry.getMemorySpaceFactory(factory_name)->create();
  }

  m_default_space = m_spaces["HOST"];
}

std::vector<std::string>
ResourceManager::getAvailableSpaces()
{
  umpire::space::MemorySpaceRegistry& registry =
    umpire::space::MemorySpaceRegistry::getInstance();

  return registry.getMemorySpaceFactoryNames();
}

void* ResourceManager::allocate(size_t bytes) {
  return allocate(bytes, m_default_space);
}

void* ResourceManager::allocate(size_t bytes, space::MemorySpace* space)
{
  void* ptr = space->allocate(bytes);
  m_allocation_spaces[ptr] = space;
  return ptr;
}

void ResourceManager::free(void* pointer)
{
  m_allocation_spaces[pointer]->free(pointer);
}

void 
  ResourceManager::setDefaultSpace(space::MemorySpace* space)
{
  m_default_space = space;
}

space::MemorySpace& ResourceManager::getDefaultSpace()
{
  return *m_default_space;
}

void ResourceManager::move(void* pointer, space::MemorySpace& destination)
{
}

} // end of namespace umpire

#include "umpire/ResourceManager.hpp"

#include "umpire/AllocatorRegistry.hpp"

#include "umpire/util/Macros.hpp"

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
  m_allocators(),
  m_allocation_to_allocator()
{
  AllocatorRegistry& registry =
    AllocatorRegistry::getInstance();
}

std::shared_ptr<Allocator>
ResourceManager::getAllocator(const std::string& name)
{
  return nullptr;
}

void 
ResourceManager::setDefaultAllocator(std::shared_ptr<Allocator> space)
{
  m_default_space = space;
}

std::shared_ptr<Allocator> ResourceManager::getDefaultAllocator()
{
  return m_default_space;
}

void ResourceManager::registerAllocation(void* ptr, std::shared_ptr<Allocator> space)
{
  m_allocation_to_allocator[ptr] = space;
}

void ResourceManager::deregisterAllocation(void* ptr)
{
  m_allocation_to_allocator.erase(ptr);
}

} // end of namespace umpire

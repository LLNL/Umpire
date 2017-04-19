#include "ResourceManager.hpp"
#include "HostSpace.hpp"

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

  MemorySpace* space = new HostSpace();
  m_spaces.push_back(new HostSpace());
  m_default_space = space;
}

std::vector<MemorySpace*>
ResourceManager::getAvailableSpaces()
{
  return m_spaces;
}

void* ResourceManager::allocate(size_t bytes) {
  return allocate(bytes, m_default_space);
}

void* ResourceManager::allocate(size_t bytes, MemorySpace* space)
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
  ResourceManager::setDefaultSpace(MemorySpace* space)
{
  m_default_space = space;
}

MemorySpace& ResourceManager::getDefaultSpace()
{
  return *m_default_space;
}

void ResourceManager::move(void* pointer, MemorySpace& destination)
{
}

} // end of namespace umpire

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/AllocatorRegistry.hpp"

#include "umpire/space/HostSpaceFactory.hpp"
#if defined(ENABLE_CUDA)
#include "umpire/space/DeviceSpaceFactory.hpp"
#include "umpire/space/UnifiedMemorySpaceFactory.hpp"
#endif

#include "umpire/op/MemoryOperationRegistry.hpp"

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
  m_allocator_names(),
  m_allocators(),
  m_allocation_to_allocator()
{
  AllocatorRegistry& registry =
    AllocatorRegistry::getInstance();

  registry.registerAllocator(
      std::make_shared<space::HostSpaceFactory>());

#if defined(ENABLE_CUDA)
  registry.registerAllocator(
      std::make_shared<space::DeviceSpaceFactory>());

  registry.registerAllocator(
      std::make_shared<space::UnifiedMemorySpaceFactory>());
#endif
}

Allocator
ResourceManager::getAllocator(const std::string& name)
{
  AllocatorRegistry& registry =
    AllocatorRegistry::getInstance();

  auto allocator = m_allocators.find(name);
  if (allocator == m_allocators.end()) {
    m_allocators[name] = registry.makeAllocator(name);
  }

  return Allocator(m_allocators[name]);
}

// void 
// ResourceManager::setDefaultAllocator(std::shared_ptr<Allocator>& allocator)
// {
//   m_default_allocator = allocator;
// }
// 
// std::shared_ptr<Allocator> ResourceManager::getDefaultAllocator()
// {
//   return m_default_allocator;
// }

void ResourceManager::registerAllocation(void* ptr, std::shared_ptr<AllocatorInterface> space)
{
  m_allocation_to_allocator[ptr] = space;
}

void ResourceManager::deregisterAllocation(void* ptr)
{
  m_allocation_to_allocator.erase(ptr);
}

void ResourceManager::copy(void* src_ptr, void* dst_ptr)
{
  auto op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc = m_allocation_to_allocator[src_ptr];
  auto dst_alloc = m_allocation_to_allocator[dst_ptr];

  std::size_t size = src_alloc->size(src_ptr);

  auto op = op_registry.find("COPY", src_alloc, dst_alloc);

  op->operator()(const_cast<const void*>(src_ptr), dst_ptr, size);
}

} // end of namespace umpire

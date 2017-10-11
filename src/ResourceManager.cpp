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

void ResourceManager::registerAllocation(void* ptr, std::shared_ptr<AllocatorInterface> space)
{
  UMPIRE_LOG("Registering " << ptr << " to " << space << " with rm " << this);
  m_allocation_to_allocator[ptr] = space;

}

void ResourceManager::deregisterAllocation(void* ptr)
{
  UMPIRE_LOG("Deregistering " << ptr);
  m_allocation_to_allocator.erase(ptr);
}

void ResourceManager::copy(void* src_ptr, void* dst_ptr)
{
  UMPIRE_LOG("Copying " << src_ptr << " to " << dst_ptr << " with rm @" << this);

  auto op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc = findAllocatorForPointer(src_ptr);
  auto dst_alloc = findAllocatorForPointer(dst_ptr);

  std::size_t src_size = src_alloc->size(src_ptr);
  std::size_t dst_size = dst_alloc->size(dst_ptr);

  if (src_size > dst_size) {
    UMPIRE_ERROR("Not enough space in destination for copy: " << src_size << " -> " << dst_size);
  }

  auto op = op_registry.find("COPY", src_alloc, dst_alloc);

  op->operator()(const_cast<const void*>(src_ptr), dst_ptr, src_size);
}

void ResourceManager::deallocate(void* ptr)
{
  auto allocator = findAllocatorForPointer(ptr);;

  allocator->deallocate(ptr);
}

std::shared_ptr<AllocatorInterface> ResourceManager::findAllocatorForPointer(void* ptr)
{
  auto allocator = m_allocation_to_allocator.find(ptr);

  if (allocator == m_allocation_to_allocator.end()) {
    UMPIRE_ERROR("Cannot find allocator " << ptr);
  }

  return allocator->second;
}

} // end of namespace umpire

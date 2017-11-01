#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationStrategyRegistry.hpp"

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
  strategy::AllocationStrategyRegistry& registry =
    strategy::AllocationStrategyRegistry::getInstance();

  registry.registerAllocationStrategy(
      std::make_shared<space::HostSpaceFactory>());

#if defined(ENABLE_CUDA)
  registry.registerAllocationStrategy(
      std::make_shared<space::DeviceSpaceFactory>());

  registry.registerAllocationStrategy(
      std::make_shared<space::UnifiedMemorySpaceFactory>());
#endif
}

std::shared_ptr<strategy::AllocationStrategy>&
ResourceManager::getAllocationStrategy(const std::string& name)
{
  strategy::AllocationStrategyRegistry& registry =
    strategy::AllocationStrategyRegistry::getInstance();

  auto allocator = m_allocators.find(name);
  if (allocator == m_allocators.end()) {
    m_allocators[name] = registry.makeAllocationStrategy(name);
  }

  return m_allocators[name];
}

Allocator
ResourceManager::getAllocator(const std::string& name)
{
  return Allocator(getAllocationStrategy(name));
}

Allocator
ResourceManager::getAllocator(void* ptr)
{
  return Allocator(findAllocatorForPointer(ptr));
}

void ResourceManager::registerAllocation(void* ptr, std::shared_ptr<strategy::AllocationStrategy> space)
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

  std::size_t src_size = src_alloc->getSize(src_ptr);
  std::size_t dst_size = dst_alloc->getSize(dst_ptr);

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

std::shared_ptr<strategy::AllocationStrategy>& ResourceManager::findAllocatorForPointer(void* ptr)
{
  auto allocator = m_allocation_to_allocator.find(ptr);

  if (allocator == m_allocation_to_allocator.end()) {
    UMPIRE_ERROR("Cannot find allocator " << ptr);
  }

  return allocator->second;
}

} // end of namespace umpire

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/resource/MemoryResourceRegistry.hpp"
#include "umpire/strategy/AllocationStrategyRegistry.hpp"

#include "umpire/strategy/AllocationStrategyFactory.hpp"

#include "umpire/resource/HostResourceFactory.hpp"
#if defined(ENABLE_CUDA)
#include "umpire/resource/DeviceResourceFactory.hpp"
#include "umpire/resource/UnifiedMemoryResourceFactory.hpp"
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

  UMPIRE_LOG(Debug, "() returning " << s_resource_manager_instance);
  return *s_resource_manager_instance;
}

ResourceManager::ResourceManager() :
  m_allocator_names(),
  m_allocators(),
  m_allocations(),
  m_memory_resources()
{
  UMPIRE_LOG(Debug, "() entering");
  resource::MemoryResourceRegistry& registry =
    resource::MemoryResourceRegistry::getInstance();

  registry.registerMemoryResource(
      std::make_shared<resource::HostResourceFactory>());

#if defined(ENABLE_CUDA)
  registry.registerMemoryResource(
    std::make_shared<resource::DeviceResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::UnifiedMemoryResourceFactory>());
#endif

  initialize();
  UMPIRE_LOG(Debug, "() leaving");
}

void
ResourceManager::initialize()
{
  UMPIRE_LOG(Debug, "() entering");
  resource::MemoryResourceRegistry& registry =
    resource::MemoryResourceRegistry::getInstance();

  m_memory_resources["HOST"] = registry.makeMemoryResource("HOST");

#if defined(ENABLE_CUDA)
  m_memory_resources["DEVICE"] = registry.makeMemoryResource("DEVICE");
  m_memory_resources["UM"] = registry.makeMemoryResource("UM");
#endif

  /*
   * Construct default allocators for each resource
   */
  m_allocators["HOST"] = m_memory_resources["HOST"];

#if defined(ENABLE_CUDA)
  /*
   *  strategy::AllocationStrategyRegistry& strategy_registry =
   *    strategy::AllocationStrategyRegistry::getInstance();
   *
   *  m_allocators["DEVICE"] = strategy_registry.makeAllocationStrategy("POOL", {}, {m_memory_resources["DEVICE"]});
   */

  m_allocators["DEVICE"] = m_memory_resources["DEVICE"];

  m_allocators["UM"] = m_memory_resources["UM"];
#endif
  UMPIRE_LOG(Debug, "() leaving");
}

std::shared_ptr<strategy::AllocationStrategy>&
ResourceManager::getAllocationStrategy(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  auto allocator = m_allocators.find(name);
  if (allocator == m_allocators.end()) {
    UMPIRE_ERROR("Allocator \"" << name << "\" not found.");
  }

  return m_allocators[name];
}

Allocator
ResourceManager::getAllocator(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  return Allocator(getAllocationStrategy(name));
}

Allocator
ResourceManager::makeAllocator(
    const std::string& name, 
    const std::string& strategy, 
    util::AllocatorTraits traits,
    std::vector<Allocator> providers)
{
  UMPIRE_LOG(Debug, "(name=\"" << name << "\", strategy=\"" << strategy << "\")");
  strategy::AllocationStrategyRegistry& registry =
    strategy::AllocationStrategyRegistry::getInstance();

  /* 
   * Turn the vector of Allocators into a vector of AllocationStrategies.
   */
  std::vector<std::shared_ptr<strategy::AllocationStrategy> > provider_strategies;
  for (auto provider : providers) {
    provider_strategies.push_back(provider.getAllocationStrategy());
  }

  m_allocators[name] = registry.makeAllocationStrategy(strategy, traits, provider_strategies);

  return Allocator(m_allocators[name]);
}

Allocator
ResourceManager::getAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return Allocator(findAllocatorForPointer(ptr));
}

void ResourceManager::registerAllocation(void* ptr, util::AllocationRecord* record)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", record=" << record << ") with " << this );

  m_allocations.insert(ptr, record);
}

void ResourceManager::deregisterAllocation(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  m_allocations.remove(ptr);
}

void ResourceManager::copy(void* src_ptr, void* dst_ptr, size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", dst_ptr=" << dst_ptr << ", size=" << size << ")");

  auto op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc_record = m_allocations.find(src_ptr);
  auto dst_alloc_record = m_allocations.find(dst_ptr);

  std::size_t src_size = src_alloc_record->m_size;
  std::size_t dst_size = dst_alloc_record->m_size;

  if (size == 0) {

    if (src_size > dst_size) {
      UMPIRE_ERROR("Not enough resource in destination for copy: " << src_size << " -> " << dst_size);
    }

    size = src_size;
  }

  auto op = op_registry.find("COPY", 
      src_alloc_record->m_strategy, 
      dst_alloc_record->m_strategy);

  op->transform(&src_ptr, &dst_ptr, size);
}

void ResourceManager::memset(void* ptr, int value, size_t length)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", value=" << value << ", length=" << length << ")");

  auto op_registry = op::MemoryOperationRegistry::getInstance();

  auto alloc_record = m_allocations.find(ptr);

  std::size_t src_size = alloc_record->m_size;

  if (length == 0) {
    length = src_size;
  }

  auto op = op_registry.find("MEMSET", 
      alloc_record->m_strategy, 
      alloc_record->m_strategy);

  op->apply(&ptr, length, value);
}

void ResourceManager::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  auto allocator = findAllocatorForPointer(ptr);;

  allocator->deallocate(ptr);
}

size_t
ResourceManager::getSize(void* ptr)
{
  auto record = m_allocations.find(ptr);
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << record->m_size);
  return record->m_size;
}

std::shared_ptr<strategy::AllocationStrategy>& ResourceManager::findAllocatorForPointer(void* ptr)
{
  auto allocation_record = m_allocations.find(ptr);

  if (! allocation_record->m_strategy) {
    UMPIRE_ERROR("Cannot find allocator " << ptr);
  }

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << allocation_record->m_strategy);
  return allocation_record->m_strategy;
}

std::vector<std::string>
ResourceManager::getAvailableAllocators()
{
  std::vector<std::string> names;
  for(auto it = m_allocators.begin(); it != m_allocators.end(); ++it) {
    names.push_back(it->first);
  }

  UMPIRE_LOG(Debug, "() returning " << names.size() << " allocators");
  return names;
}

} // end of namespace umpire

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/resource/HostResourceFactory.hpp"
#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/resource/DeviceResourceFactory.hpp"
#include "umpire/resource/UnifiedMemoryResourceFactory.hpp"
#include "umpire/resource/PinnedMemoryResourceFactory.hpp"
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
  m_allocators_by_name(),
  m_allocators_by_id(),
  m_allocations(),
  m_memory_resources(),
  m_id(0)
{
  UMPIRE_LOG(Debug, "() entering");
  resource::MemoryResourceRegistry& registry =
    resource::MemoryResourceRegistry::getInstance();

  registry.registerMemoryResource(
      std::make_shared<resource::HostResourceFactory>());

#if defined(UMPIRE_ENABLE_CUDA)
  registry.registerMemoryResource(
    std::make_shared<resource::DeviceResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::UnifiedMemoryResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::PinnedMemoryResourceFactory>());
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

  m_memory_resources[resource::Host] = registry.makeMemoryResource("HOST", getNextId());

#if defined(UMPIRE_ENABLE_CUDA)
  m_memory_resources[resource::Device] = registry.makeMemoryResource("DEVICE", getNextId());
  m_memory_resources[resource::UnifiedMemory] = registry.makeMemoryResource("UM", getNextId());
  m_memory_resources[resource::PinnedMemory] = registry.makeMemoryResource("PINNED", getNextId());
#endif

  /*
   * Construct default allocators for each resource
   */
  auto host_allocator = m_memory_resources[resource::Host];
  m_allocators_by_name["HOST"] = host_allocator;
  m_allocators_by_id[host_allocator->getId()] = host_allocator;

#if defined(UMPIRE_ENABLE_CUDA)
  /*
   *  strategy::AllocationStrategyRegistry& strategy_registry =
   *    strategy::AllocationStrategyRegistry::getInstance();
   *
   *  m_allocators_by_name["DEVICE"] = strategy_registry.makeAllocationStrategy("POOL", {}, {m_memory_resources["DEVICE"]});
   */

  auto device_allocator = m_memory_resources[resource::Device];
  m_allocators_by_name["DEVICE"] = device_allocator;
  m_allocators_by_id[device_allocator->getId()] = device_allocator;

  auto um_allocator = m_memory_resources[resource::UnifiedMemory];
  m_allocators_by_name["UM"] = um_allocator;
  m_allocators_by_id[um_allocator->getId()] = um_allocator;

  auto pinned_allocator = m_memory_resources[resource::PinnedMemory];
  m_allocators_by_name["PINNED"] = pinned_allocator;
  m_allocators_by_id[pinned_allocator->getId()] = pinned_allocator;
#endif
  UMPIRE_LOG(Debug, "() leaving");
}

std::shared_ptr<strategy::AllocationStrategy>&
ResourceManager::getAllocationStrategy(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  auto allocator = m_allocators_by_name.find(name);
  if (allocator == m_allocators_by_name.end()) {
    UMPIRE_ERROR("Allocator \"" << name << "\" not found.");
  }

  return m_allocators_by_name[name];
}

Allocator
ResourceManager::getAllocator(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  return Allocator(getAllocationStrategy(name));
}

Allocator
ResourceManager::getAllocator(resource::MemoryResourceType resource_type)
{
  UMPIRE_LOG(Debug, "(\"" << static_cast<size_t>(resource_type) << "\")");

  auto allocator = m_memory_resources.find(resource_type);
  if (allocator == m_memory_resources.end()) {
    UMPIRE_ERROR("Allocator \"" << static_cast<size_t>(resource_type) << "\" not found.");
  }

  return Allocator(m_memory_resources[resource_type]);
}

Allocator
ResourceManager::getAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return Allocator(findAllocatorForPointer(ptr));
}

bool
ResourceManager::isAllocator(const std::string& name)
{
  return (m_allocators_by_name.find(name) != m_allocators_by_name.end());
}

bool
ResourceManager::hasAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr <<")");

  return m_allocations.contains(ptr);
}

void ResourceManager::registerAllocation(void* ptr, util::AllocationRecord* record)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", record=" << record << ") with " << this );

  m_allocations.insert(ptr, record);
}

util::AllocationRecord* ResourceManager::deregisterAllocation(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return m_allocations.remove(ptr);
}

bool
ResourceManager::isAllocatorRegistered(const std::string& name)
{
  return (m_allocators_by_name.find(name) != m_allocators_by_name.end());
}

void ResourceManager::copy(void* dst_ptr, void* src_ptr, size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", dst_ptr=" << dst_ptr << ", size=" << size << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

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

  op->transform(src_ptr, &dst_ptr, src_alloc_record, dst_alloc_record, size);
}

void ResourceManager::memset(void* ptr, int value, size_t length)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", value=" << value << ", length=" << length << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto alloc_record = m_allocations.find(ptr);

  std::size_t src_size = alloc_record->m_size;

  if (length == 0) {
    length = src_size;
  }

  auto op = op_registry.find("MEMSET", 
      alloc_record->m_strategy, 
      alloc_record->m_strategy);

  op->apply(ptr, alloc_record, value, length);
}

void*
ResourceManager::reallocate(void* src_ptr, size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", size=" << size << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto alloc_record = m_allocations.find(src_ptr);

  if (src_ptr != alloc_record->m_ptr) {
    UMPIRE_ERROR("Cannot reallocate an offset ptr (ptr=" << src_ptr << ", base=" << alloc_record->m_ptr);
  }

  auto op = op_registry.find("REALLOCATE", 
      alloc_record->m_strategy, 
      alloc_record->m_strategy);

  void* dst_ptr = nullptr;

  op->transform(src_ptr, &dst_ptr, alloc_record, alloc_record, size);

  return dst_ptr;
}

void*
ResourceManager::move(void* ptr, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << ptr << ", allocator=" << allocator.getName() << ")");

  auto alloc_record = m_allocations.find(ptr);

  // short-circuit if ptr was allocated by 'allocator'
  if (alloc_record->m_strategy == allocator.getAllocationStrategy()) {
    return ptr;
  }

  if (ptr != alloc_record->m_ptr) {
    UMPIRE_ERROR("Cannot move an offset ptr (ptr=" << ptr << ", base=" << alloc_record->m_ptr);
  }

  size_t size = alloc_record->m_size;
  void* dst_ptr = allocator.allocate(size);

  copy(dst_ptr, ptr);

  deallocate(ptr);

  return dst_ptr;
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
  for(auto it = m_allocators_by_name.begin(); it != m_allocators_by_name.end(); ++it) {
    names.push_back(it->first);
  }

  UMPIRE_LOG(Debug, "() returning " << names.size() << " allocators");
  return names;
}

int
ResourceManager::getNextId()
{
  return m_id++;
}

} // end of namespace umpire

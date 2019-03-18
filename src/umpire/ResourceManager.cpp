//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#include <memory>
#include <iostream>

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#endif

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>

#include "umpire/resource/CudaDeviceResourceFactory.hpp"
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"
#include "umpire/resource/CudaPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_ROCM)
#include "umpire/resource/RocmDeviceResourceFactory.hpp"
#include "umpire/resource/RocmPinnedMemoryResourceFactory.hpp"
#endif

#include "umpire/op/MemoryOperationRegistry.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationTracker.hpp"

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
  m_default_allocator(),
  m_memory_resources(),
  m_id(0),
  m_mutex(new std::mutex())
{
  UMPIRE_LOG(Debug, "() entering");
  resource::MemoryResourceRegistry& registry =
    resource::MemoryResourceRegistry::getInstance();

  registry.registerMemoryResource(
      std::make_shared<resource::HostResourceFactory>());

#if defined(UMPIRE_ENABLE_CUDA)
  registry.registerMemoryResource(
    std::make_shared<resource::CudaDeviceResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::CudaUnifiedMemoryResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::CudaPinnedMemoryResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::CudaConstantMemoryResourceFactory>());
#endif

#if defined(UMPIRE_ENABLE_ROCM)
  registry.registerMemoryResource(
    std::make_shared<resource::RocmDeviceResourceFactory>());

  registry.registerMemoryResource(
    std::make_shared<resource::RocmPinnedMemoryResourceFactory>());
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
  int count;
  auto error = ::cudaGetDeviceCount(&count);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("Umpire compiled with CUDA support but no GPUs detected!");
  }
#endif

#if defined(UMPIRE_ENABLE_DEVICE)
  m_memory_resources[resource::Device] = registry.makeMemoryResource("DEVICE", getNextId());
#endif

#if defined(UMPIRE_ENABLE_PINNED)
  m_memory_resources[resource::Pinned] = registry.makeMemoryResource("PINNED", getNextId());
#endif

#if defined(UMPIRE_ENABLE_UM)
  m_memory_resources[resource::Unified] = registry.makeMemoryResource("UM", getNextId());
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  m_memory_resources[resource::Constant] = registry.makeMemoryResource("DEVICE_CONST", getNextId());
#endif

  /*
   * Construct default allocators for each resource
   */
  auto host_allocator = m_memory_resources[resource::Host];
  m_allocators_by_name["HOST"] = host_allocator;
  m_allocators_by_id[host_allocator->getId()] = host_allocator;

  m_default_allocator = host_allocator;

#if defined(UMPIRE_ENABLE_DEVICE)
  auto device_allocator = m_memory_resources[resource::Device];
  m_allocators_by_name["DEVICE"] = device_allocator;
  m_allocators_by_id[device_allocator->getId()] = device_allocator;
#endif

#if defined(UMPIRE_ENABLE_PINNED)
  auto pinned_allocator = m_memory_resources[resource::Pinned];
  m_allocators_by_name["PINNED"] = pinned_allocator;
  m_allocators_by_id[pinned_allocator->getId()] = pinned_allocator;
#endif

#if defined(UMPIRE_ENABLE_UM)
  auto um_allocator = m_memory_resources[resource::Unified];
  m_allocators_by_name["UM"] = um_allocator;
  m_allocators_by_id[um_allocator->getId()] = um_allocator;
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  auto device_const_allocator = m_memory_resources[resource::Constant];
  m_allocators_by_name["DEVICE_CONST"] = device_const_allocator;
  m_allocators_by_id[device_const_allocator->getId()] = device_const_allocator;
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
ResourceManager::getAllocator(int id)
{
  UMPIRE_LOG(Debug, "(\"" << id << "\")");

  auto allocator = m_allocators_by_id.find(id);
  if (allocator == m_allocators_by_id.end()) {
    UMPIRE_ERROR("Allocator \"" << id << "\" not found.");
  }

  return Allocator(m_allocators_by_id[id]);
}

Allocator
ResourceManager::getDefaultAllocator()
{
  UMPIRE_LOG(Debug, "");

  if (!m_default_allocator) {
    UMPIRE_ERROR("The default Allocator is not defined");
  }

  return Allocator(m_default_allocator);
}

void
ResourceManager::setDefaultAllocator(Allocator allocator) noexcept
{
  UMPIRE_LOG(Debug, "(\"" << allocator.getName() << "\")");

  m_default_allocator = allocator.getAllocationStrategy();
}

void
ResourceManager::registerAllocator(const std::string& name, Allocator allocator)
{
  if (isAllocator(name)) {
    UMPIRE_ERROR("Allocator " << name << " is already registered.");
  }

  m_allocators_by_name[name] = allocator.getAllocationStrategy();
}

Allocator
ResourceManager::getAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return Allocator(findAllocatorForPointer(ptr));
}

bool
ResourceManager::isAllocator(const std::string& name) noexcept
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

const util::AllocationRecord*
ResourceManager::findAllocationRecord(void* ptr) const
{
  auto alloc_record = m_allocations.find(ptr);

  if (!alloc_record->m_strategy) {
    UMPIRE_ERROR("Cannot find allocator for " << ptr);
  }

  UMPIRE_LOG(Debug, "(Returning allocation record for ptr = " << ptr << ")");

  return alloc_record;
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
    size = src_size;
  }

  if (size > dst_size) {
    UMPIRE_ERROR("Not enough resource in destination for copy: " << size << " -> " << dst_size);
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

  if (length > src_size) {
    UMPIRE_ERROR("Cannot memset over the end of allocation: " << length << " -> " << src_size);
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

  void* dst_ptr = nullptr;

  if (!src_ptr) {
    dst_ptr = m_default_allocator->allocate(size);
  } else {
    auto& op_registry = op::MemoryOperationRegistry::getInstance();

    auto alloc_record = m_allocations.find(src_ptr);

    if (src_ptr != alloc_record->m_ptr) {
      UMPIRE_ERROR("Cannot reallocate an offset ptr (ptr=" << src_ptr << ", base=" << alloc_record->m_ptr);
    }

    auto op = op_registry.find("REALLOCATE",
        alloc_record->m_strategy,
        alloc_record->m_strategy);


    op->transform(src_ptr, &dst_ptr, alloc_record, alloc_record, size);
  }

  return dst_ptr;
}

void*
ResourceManager::reallocate(void* src_ptr, size_t size, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", size=" << size << ")");

  void* dst_ptr = nullptr;

  if (!src_ptr) {
    dst_ptr = allocator.allocate(size);
  } else {
    auto alloc_record = m_allocations.find(src_ptr);

    if (alloc_record->m_strategy == allocator.getAllocationStrategy()) {
      dst_ptr = reallocate(src_ptr, size);
    } else {
      UMPIRE_ERROR("Cannot reallocate " << src_ptr << " with Allocator " << allocator.getName());
    }
  }

  return dst_ptr;
}

#if defined(UMPIRE_ENABLE_NUMA)
static std::shared_ptr<strategy::NumaPolicy> cast_as_numa_policy(Allocator& allocator) {
  std::shared_ptr<strategy::NumaPolicy> numa_alloc;

  numa_alloc = std::dynamic_pointer_cast<strategy::NumaPolicy>(
    allocator.getAllocationStrategy());

  // ... or an AllocationTracker wrapping a NumaPolicy
  if (!numa_alloc) {
    auto alloc_tracker = std::dynamic_pointer_cast<strategy::AllocationTracker>(
      allocator.getAllocationStrategy());
    if (alloc_tracker) {
      numa_alloc = std::dynamic_pointer_cast<strategy::NumaPolicy>(
        alloc_tracker->getAllocationStrategy());
    }
  }

  return numa_alloc;
}
#endif

void*
ResourceManager::move(void* ptr, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << ptr << ", allocator=" << allocator.getName() << ")");

  auto alloc_record = m_allocations.find(ptr);

  // short-circuit if ptr was allocated by 'allocator'
  if (alloc_record->m_strategy == allocator.getAllocationStrategy()) {
    return ptr;
  }

#if defined(UMPIRE_ENABLE_NUMA)
  {
    // short circuit if allocator has a NumaPolicy
    auto numa_alloc = cast_as_numa_policy(allocator);

    // If found, use op::NumaMoveOperation to move in-place (same address returned)
    if (numa_alloc) {
      auto& op_registry = op::MemoryOperationRegistry::getInstance();

      auto src_alloc_record = m_allocations.find(ptr);

      const size_t size = src_alloc_record->m_size;
      util::AllocationRecord dst_alloc_record;
      dst_alloc_record.m_size = src_alloc_record->m_size;
      dst_alloc_record.m_strategy = numa_alloc;

      void *ret = nullptr;
      if (size > 0) {
        auto op = op_registry.find("MOVE",
                                   src_alloc_record->m_strategy,
                                   dst_alloc_record.m_strategy);

        op->transform(ptr, &ret, src_alloc_record, &dst_alloc_record, size);
        if (ret != ptr) {
          UMPIRE_ERROR("Numa move error");
        }
      }
      else {
        ret = ptr;
      }

      return ret;
    }
  }
#endif

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
ResourceManager::getSize(void* ptr) const
{
  auto record = m_allocations.find(ptr);
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << record->m_size);
  return record->m_size;
}

void
ResourceManager::coalesce(Allocator allocator)
{
  auto strategy = allocator.getAllocationStrategy();

  auto tracker = std::dynamic_pointer_cast<umpire::strategy::AllocationTracker>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();

  }

  auto dynamic_pool = std::dynamic_pointer_cast<umpire::strategy::DynamicPool>(strategy);

  if (dynamic_pool) {
    dynamic_pool->coalesce();
  } else {
    UMPIRE_ERROR(allocator.getName() << " is not a DynamicPool, cannot coalesce!");
  }
}

std::shared_ptr<strategy::AllocationStrategy>& ResourceManager::findAllocatorForId(int id)
{
  auto allocator_i = m_allocators_by_id.find(id);

  if ( allocator_i == m_allocators_by_id.end() ) {
    UMPIRE_ERROR("Cannot find allocator for ID " << id);
  }

  UMPIRE_LOG(Debug, "(id=" << id << ") returning " << allocator_i->second );
  return allocator_i->second;
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
ResourceManager::getAvailableAllocators() noexcept
{
  std::vector<std::string> names;
  for(auto it = m_allocators_by_name.begin(); it != m_allocators_by_name.end(); ++it) {
    names.push_back(it->first);
  }

  UMPIRE_LOG(Debug, "() returning " << names.size() << " allocators");
  return names;
}

int
ResourceManager::getNextId() noexcept
{
  return m_id++;
}

void print_allocator_records(Allocator alloc, std::ostream& os) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = alloc.getAllocationStrategy().get();

  rm.m_allocations.print([strategy] (const util::AllocationRecord* rec) {
    return rec->m_strategy.get() == strategy;
  }, os);
}


} // end of namespace umpire

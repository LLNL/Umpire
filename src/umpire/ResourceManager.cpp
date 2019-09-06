//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/resource/MemoryResourceRegistry.hpp"

#include "umpire/resource/HostResourceFactory.hpp"

#include "umpire/resource/NullMemoryResourceFactory.hpp"

#if defined(UMPIRE_ENABLE_SICM)
#include "umpire/resource/SICMResourceFactory.hpp"
#include "umpire/strategy/SICMStrategy.hpp"
#endif

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

#if defined(UMPIRE_ENABLE_HCC)
#include "umpire/resource/RocmDeviceResourceFactory.hpp"
#include "umpire/resource/RocmPinnedMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "umpire/resource/HipDeviceResourceFactory.hpp"
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"
#endif

#include "umpire/op/MemoryOperationRegistry.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#include "umpire/strategy/ZeroByteHandler.hpp"
#include "umpire/strategy/FixedPool.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"
#include "umpire/util/wrap_allocator.hpp"

#include "umpire/util/MPI.hpp"
#include "umpire/util/io.hpp"

#include <iterator>
#include <sstream>
#include <memory>

static const char* s_null_resource_name{"__umpire_internal_null"};
static const char* s_zero_byte_pool_name{"__umpire_internal_0_byte_pool"};

namespace umpire {

ResourceManager&
ResourceManager::getInstance()
{
  static ResourceManager resource_manager;

  UMPIRE_LOG(Debug, "() returning " << &resource_manager);
  return resource_manager;
}

ResourceManager::ResourceManager() :
  m_allocations(),
  m_allocators(),
  m_allocators_by_id(),
  m_allocators_by_name(),
  m_memory_resources(),
  m_default_allocator(),
  m_id(0),
  m_mutex()
{
  UMPIRE_LOG(Debug, "() entering");

  const char* env_enable_replay{getenv("UMPIRE_REPLAY")};
  const bool enable_replay{env_enable_replay != nullptr};

  const char* env_enable_log{getenv("UMPIRE_LOG_LEVEL")};
  const bool enable_log{env_enable_log != nullptr};

  util::initialize_io(enable_log, enable_replay);

  resource::MemoryResourceRegistry& registry{
    resource::MemoryResourceRegistry::getInstance()};

// read config file/figure out which device belongs to what

  registry.registerMemoryResource(
      util::make_unique<resource::NullMemoryResourceFactory>());

#if defined(UMPIRE_ENABLE_SICM)
  registry.registerMemoryResource(
    util::make_unique<resource::SICMResourceFactory>("HOST", {0, 1, 2, 3, 4, 5}));
#else
  registry.registerMemoryResource(
    util::make_unique<resource::HostResourceFactory>());
#endif

#if defined(UMPIRE_ENABLE_CUDA)
#if defined(UMPIRE_ENABLE_SICM)
  registry.registerMemoryResource(
    util::make_unique<resource::SICMResourceFactory>("DEVICE", {}));

  registry.registerMemoryResource(
    util::make_unique<resource::SICMResourceFactory>("UM", {}));

  registry.registerMemoryResource(
    util::make_unique<resource::SICMResourceFactory>("PINNED", {}));

  registry.registerMemoryResource(
    util::make_unique<resource::SICMResourceFactory>("CONSTANT", {}));
#else
  registry.registerMemoryResource(
    util::make_unique<resource::CudaDeviceResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::CudaUnifiedMemoryResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::CudaPinnedMemoryResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::CudaConstantMemoryResourceFactory>());
#endif
#endif

#if defined(UMPIRE_ENABLE_HCC)
#if defined(UMPIRE_ENABLE_SICM)
  registry.registerMemoryResource(
      util::make_unique<resource::SICMResourceFactory>("DEVICE", {}));

  registry.registerMemoryResource(
      util::make_unique<resource::SICMResourceFactory>("PINNED", {}));
#else
  registry.registerMemoryResource(
    util::make_unique<resource::RocmDeviceResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::RocmPinnedMemoryResourceFactory>());
#endif
#endif

#if defined(UMPIRE_ENABLE_HIP)
  registry.registerMemoryResource(
    util::make_unique<resource::HipDeviceResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::HipPinnedMemoryResourceFactory>());

  registry.registerMemoryResource(
    util::make_unique<resource::HipConstantMemoryResourceFactory>());
#endif

  initialize();

  UMPIRE_LOG(Debug, "() leaving");
}

ResourceManager::~ResourceManager()
{
  for (auto&& allocator : m_allocators) {
    allocator.reset();
  }
}

void
ResourceManager::initialize()
{
  UMPIRE_LOG(Debug, "() entering");

  UMPIRE_LOG(Debug, "Umpire v" << UMPIRE_VERSION_MAJOR << "." <<
      UMPIRE_VERSION_MINOR << "." <<
      UMPIRE_VERSION_PATCH << "." <<
      UMPIRE_VERSION_RC);

  UMPIRE_REPLAY( "\"event\": \"version\", \"payload\": { \"major\":" << UMPIRE_VERSION_MAJOR
      << ", \"minor\":" << UMPIRE_VERSION_MINOR
      << ", \"patch\":" << UMPIRE_VERSION_PATCH
      << ", \"rc\": \"" << UMPIRE_VERSION_RC << "\""
      << " }");

  resource::MemoryResourceRegistry& registry{
    resource::MemoryResourceRegistry::getInstance()};

  {
    std::unique_ptr<strategy::AllocationStrategy>
      host_allocator{
        util::wrap_allocator<
          strategy::AllocationTracker,
          strategy::ZeroByteHandler>(
            registry.makeMemoryResource("HOST", getNextId()))};

    UMPIRE_REPLAY(
         "\"event\": \"makeMemoryResource\""
      << ", \"payload\": { \"name\": \"" << "HOST" << "\" }"
      << ", \"result\": \"" << host_allocator.get() << "\""
    );

    int id{host_allocator->getId()};
    m_allocators_by_name["HOST"]  = host_allocator.get();
    m_memory_resources[resource::Host] = host_allocator.get();
    m_default_allocator = host_allocator.get();
    m_allocators_by_id[id] = host_allocator.get();
    m_allocators.emplace_front(std::move(host_allocator));
  }

  {
    std::unique_ptr<strategy::AllocationStrategy>
      allocator{
        //util::wrap_allocator<strategy::AllocationTracker>(
          registry.makeMemoryResource(s_null_resource_name, getNextId())};

    int id{allocator->getId()};
    m_allocators_by_name[s_null_resource_name]  = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }

#if defined(UMPIRE_ENABLE_SICM)
  m_memory_resources[resource::SICM] = registry.makeMemoryResource("SICM", getNextId());
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  int count;
  auto error = ::cudaGetDeviceCount(&count);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("Umpire compiled with CUDA support but no GPUs detected!");
  }
#endif

#if defined(UMPIRE_ENABLE_HIP)
  int count;
  auto error = ::hipGetDeviceCount(&count);

  if (error != hipSuccess) {
    UMPIRE_ERROR("Umpire compiled with HIP support but no GPUs detected!");
  }
#endif

#if defined(UMPIRE_ENABLE_DEVICE)
  {
    std::unique_ptr<strategy::AllocationStrategy>
      allocator{util::wrap_allocator<
        strategy::AllocationTracker,
        strategy::ZeroByteHandler>(
            registry.makeMemoryResource("DEVICE", getNextId()))};
    UMPIRE_REPLAY(
         "\"event\": \"makeMemoryResource\""
      << ", \"payload\": { \"name\": \"" << "DEVICE" << "\" }"
      << ", \"result\": \"" << allocator.get() << "\""
    );


    int id{allocator->getId()};
    m_allocators_by_name["DEVICE"] = allocator.get();
    m_memory_resources[resource::Device] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }
  m_memory_resources[resource::Device] = registry.makeMemoryResource("DEVICE", getNextId());
#endif

#if defined(UMPIRE_ENABLE_PINNED)
  m_memory_resources[resource::Pinned] = registry.makeMemoryResource("PINNED", getNextId());
#endif

#if defined(UMPIRE_ENABLE_UM)
  m_memory_resources[resource::Unified] = registry.makeMemoryResource("UM", getNextId());
#endif

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
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
  {
    std::unique_ptr<strategy::AllocationStrategy>
      allocator{util::wrap_allocator<
        strategy::AllocationTracker,
        strategy::ZeroByteHandler>(
            registry.makeMemoryResource("PINNED", getNextId()))};
    UMPIRE_REPLAY(
         "\"event\": \"makeMemoryResource\""
      << ", \"payload\": { \"name\": \"" << "PINNED" << "\" }"
      << ", \"result\": \"" << allocator.get() << "\""
    );

    int id{allocator->getId()};
    m_allocators_by_name["PINNED"] = allocator.get();
    m_memory_resources[resource::Pinned] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }
#endif

#if defined(UMPIRE_ENABLE_UM)
  {
    std::unique_ptr<strategy::AllocationStrategy>
      allocator{util::wrap_allocator<
        strategy::AllocationTracker,
        strategy::ZeroByteHandler>(
            registry.makeMemoryResource("UM", getNextId()))};
    UMPIRE_REPLAY(
         "\"event\": \"makeMemoryResource\""
      << ", \"payload\": { \"name\": \"" << "UM" << "\" }"
      << ", \"result\": \"" << allocator.get() << "\""
    );

    int id{allocator->getId()};
    m_allocators_by_name["UM"] = allocator.get();
    m_memory_resources[resource::Unified] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }
#endif

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  {
    std::unique_ptr<strategy::AllocationStrategy>
      allocator{util::wrap_allocator<
        strategy::AllocationTracker,
        strategy::ZeroByteHandler>(
            registry.makeMemoryResource("DEVICE_CONST", getNextId()))};
    UMPIRE_REPLAY(
         "\"event\": \"makeMemoryResource\""
      << ", \"payload\": { \"name\": \"" << "DEVICE_CONST" << "\" }"
      << ", \"result\": \"" << allocator.get() << "\""
    );

    int id{allocator->getId()};
    m_allocators_by_name["DEVICE_CONST"] = allocator.get();
    m_memory_resources[resource::Constant] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }
#endif

  {
    std::unique_ptr<strategy::AllocationStrategy> allocator{
      new strategy::FixedPool{s_zero_byte_pool_name,
        getNextId(),
        Allocator{m_allocators_by_name[s_null_resource_name]},
        1}
    };

    int id{allocator->getId()};
    m_allocators_by_name[s_zero_byte_pool_name] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }

  UMPIRE_LOG(Debug, "() leaving");
}

strategy::AllocationStrategy*
ResourceManager::getAllocationStrategy(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  auto allocator = m_allocators_by_name.find(name);
  if (allocator == m_allocators_by_name.end()) {
    UMPIRE_ERROR("Allocator \"" << name << "\" not found. Available allocators: "
        << getAllocatorInformation());
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
ResourceManager::getAllocator(const char* name)
{
  return getAllocator(std::string{name});
}

Allocator
ResourceManager::getAllocator(resource::MemoryResourceType resource_type)
{
  UMPIRE_LOG(Debug, "(\"" << static_cast<std::size_t>(resource_type) << "\")");

  auto allocator = m_memory_resources.find(resource_type);
  if (allocator == m_memory_resources.end()) {
    UMPIRE_ERROR("Allocator \"" << static_cast<std::size_t>(resource_type)
        << "\" not found. Available allocators: " << getAllocatorInformation());
  }

  return Allocator(m_memory_resources[resource_type]);
}

Allocator
ResourceManager::getAllocator(int id)
{
  UMPIRE_LOG(Debug, "(\"" << id << "\")");

  auto allocator = m_allocators_by_id.find(id);
  if (allocator == m_allocators_by_id.end()) {
    UMPIRE_ERROR("Allocator \"" << id << "\" not found. Available allocators: "
        << getAllocatorInformation());
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

void ResourceManager::registerAllocation(void* ptr, util::AllocationRecord record)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", size=" << record.size
             << ", strategy=" << record.strategy << ") with " << this);
  m_allocations.insert(ptr, record);
}

util::AllocationRecord ResourceManager::deregisterAllocation(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return m_allocations.remove(ptr);
}

const util::AllocationRecord*
ResourceManager::findAllocationRecord(void* ptr) const
{
  auto alloc_record = m_allocations.find(ptr);

  if (!alloc_record->strategy) {
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

void ResourceManager::copy(void* dst_ptr, void* src_ptr, std::size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", dst_ptr=" << dst_ptr << ", size=" << size << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc_record = m_allocations.find(src_ptr);
  std::ptrdiff_t src_offset = static_cast<char*>(src_ptr) - static_cast<char*>(src_alloc_record->ptr);
  std::size_t src_size = src_alloc_record->size - src_offset;

  auto dst_alloc_record = m_allocations.find(dst_ptr);
  std::ptrdiff_t dst_offset = static_cast<char*>(dst_ptr) - static_cast<char*>(dst_alloc_record->ptr);
  std::size_t dst_size = dst_alloc_record->size - dst_offset;

  if (size == 0) {
    size = src_size;
  }

  if (size > dst_size) {
    UMPIRE_ERROR("Not enough resource in destination for copy: " << size << " -> " << dst_size);
  }

  auto op = op_registry.find("COPY",
      src_alloc_record->strategy,
      dst_alloc_record->strategy);

  op->transform(src_ptr, &dst_ptr, src_alloc_record, dst_alloc_record, size);
}

void ResourceManager::memset(void* ptr, int value, std::size_t length)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", value=" << value << ", length=" << length << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto alloc_record = m_allocations.find(ptr);

  std::ptrdiff_t offset = static_cast<char*>(ptr) - static_cast<char*>(alloc_record->ptr);
  std::size_t size = alloc_record->size - offset;

  if (length == 0) {
    length = size;
  }

  if (length > size) {
    UMPIRE_ERROR("Cannot memset over the end of allocation: " << length << " -> " << size);
  }

  auto op = op_registry.find("MEMSET",
      alloc_record->strategy,
      alloc_record->strategy);

  op->apply(ptr, alloc_record, value, length);
}

void*
ResourceManager::reallocate(void* src_ptr, std::size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", size=" << size << ")");

  void* dst_ptr = nullptr;

  if (!src_ptr) {
    dst_ptr = m_default_allocator->allocate(size);
  } else {
    auto& op_registry = op::MemoryOperationRegistry::getInstance();

    auto alloc_record = m_allocations.find(src_ptr);

    if (src_ptr != alloc_record->ptr) {
      UMPIRE_ERROR("Cannot reallocate an offset ptr (ptr=" << src_ptr << ", base=" << alloc_record->ptr);
    }

    auto op = op_registry.find("REALLOCATE",
        alloc_record->strategy,
        alloc_record->strategy);


    op->transform(src_ptr, &dst_ptr, alloc_record, alloc_record, size);
  }

  return dst_ptr;
}

void*
ResourceManager::reallocate(void* src_ptr, std::size_t size, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", size=" << size << ")");

  void* dst_ptr = nullptr;

  if (!src_ptr) {
    dst_ptr = allocator.allocate(size);
  } else {
    auto alloc_record = m_allocations.find(src_ptr);

    if (alloc_record->strategy == allocator.getAllocationStrategy()) {
      dst_ptr = reallocate(src_ptr, size);
    } else {
      UMPIRE_ERROR("Cannot reallocate " << src_ptr << " with Allocator " << allocator.getName());
    }
  }

  return dst_ptr;
}

void*
ResourceManager::move(void* ptr, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << ptr << ", allocator=" << allocator.getName() << ")");

  auto alloc_record = m_allocations.find(ptr);

  // short-circuit if ptr was allocated by 'allocator'
  if (alloc_record->strategy == allocator.getAllocationStrategy()) {
    return ptr;
  }

#if defined(UMPIRE_ENABLE_SICM)
  auto sicm_alloc = static_cast<strategy::SICMStrategy *>(allocator.m_allocator);
  if (sicm_alloc) {
      auto& op_registry = op::MemoryOperationRegistry::getInstance();

      auto src_alloc_record = m_allocations.find(ptr);

      const size_t size = src_alloc_record->m_size;
      util::AllocationRecord dst_alloc_record;
      dst_alloc_record.m_size = src_alloc_record->m_size;
      dst_alloc_record.m_strategy = sicm_alloc;

      void *ret = nullptr;
      if (size > 0) {
        auto op = op_registry.find("MOVE",
                                   src_alloc_record->m_strategy,
                                   dst_alloc_record.m_strategy);

        op->transform(ptr, &ret, src_alloc_record, &dst_alloc_record, size);
        if (ret != ptr) {
          UMPIRE_ERROR("SICM move error");
        }
      }
      else {
        ret = ptr;
      }

      return ret;
  }
#endif

#if defined(UMPIRE_ENABLE_NUMA)
  {
    auto base_strategy = util::unwrap_allocator<strategy::AllocationStrategy>(allocator);

    // If found, use op::NumaMoveOperation to move in-place (same address returned)
    if (dynamic_cast<strategy::NumaPolicy*>(base_strategy)) {
      auto& op_registry = op::MemoryOperationRegistry::getInstance();

      auto src_alloc_record = m_allocations.find(ptr);

      const std::size_t size{src_alloc_record->size};
      util::AllocationRecord dst_alloc_record{
        nullptr, size, allocator.getAllocationStrategy()};

      if (size > 0) {
        auto op = op_registry.find("MOVE",
                                   src_alloc_record->strategy,
                                   dst_alloc_record.strategy);
        void *ret{nullptr};
        op->transform(ptr, &ret, src_alloc_record, &dst_alloc_record, size);
        UMPIRE_ASSERT(ret == ptr);
      }

      return ptr;
    }
  }
#endif

  if (ptr != alloc_record->ptr) {
    UMPIRE_ERROR("Cannot move an offset ptr (ptr=" << ptr << ", base=" << alloc_record->ptr);
  }

  void* dst_ptr{allocator.allocate(alloc_record->size)};
  copy(dst_ptr, ptr);

  deallocate(ptr);

  return dst_ptr;
}

void ResourceManager::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  auto allocator = findAllocatorForPointer(ptr);

  allocator->deallocate(ptr);
}

std::size_t
ResourceManager::getSize(void* ptr) const
{
  auto record = m_allocations.find(ptr);
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << record->size);
  return record->size;
}

strategy::AllocationStrategy* ResourceManager::findAllocatorForId(int id)
{
  auto allocator_i = m_allocators_by_id.find(id);

  if ( allocator_i == m_allocators_by_id.end() ) {
    UMPIRE_ERROR("Cannot find allocator for ID " << id);
  }

  UMPIRE_LOG(Debug, "(id=" << id << ") returning " << allocator_i->second );
  return allocator_i->second;
}

strategy::AllocationStrategy* ResourceManager::findAllocatorForPointer(void* ptr)
{
  auto allocation_record = m_allocations.find(ptr);

  if (! allocation_record->strategy) {
    UMPIRE_ERROR("Cannot find allocator " << ptr);
  }

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << allocation_record->strategy);
  return allocation_record->strategy;
}

std::vector<std::string>
ResourceManager::getAllocatorNames() const noexcept
{
  std::vector<std::string> names;
  for(auto it = m_allocators_by_name.begin(); it != m_allocators_by_name.end(); ++it) {
    names.push_back(it->first);
  }

  UMPIRE_LOG(Debug, "() returning " << names.size() << " allocators");
  return names;
}

std::vector<int>
ResourceManager::getAllocatorIds() const noexcept
{
  std::vector<int> ids;
  for (auto& it : m_allocators_by_id) {
    ids.push_back(it.first);
  }

  return ids;
}

int
ResourceManager::getNextId() noexcept
{
  return m_id++;
}

std::string
ResourceManager::getAllocatorInformation() const noexcept
{
  std::ostringstream info;

  for (auto& it : m_allocators_by_name) {
    info << *it.second << " ";
  }

  return info.str();
}

strategy::AllocationStrategy*
ResourceManager::getZeroByteAllocator()
{
  return m_allocators_by_name[s_zero_byte_pool_name];
}

} // end of namespace umpire

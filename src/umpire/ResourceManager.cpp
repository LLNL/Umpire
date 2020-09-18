//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/ResourceManager.hpp"

#include "umpire/config.hpp"
#include "umpire/resource/HostResourceFactory.hpp"
#include "umpire/resource/MemoryResourceRegistry.hpp"
#include "umpire/resource/NullMemoryResourceFactory.hpp"

#if defined(UMPIRE_ENABLE_BOOST_RESOURCE)
#include "umpire/resource/BoostMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
#include "umpire/resource/FileMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#endif

#if defined(UMPIRE_ENABLE_CUDA)
#include <cuda_runtime_api.h>

#include "umpire/resource/CudaDeviceResourceFactory.hpp"
#include "umpire/resource/CudaPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"
#endif
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "umpire/resource/HipDeviceResourceFactory.hpp"
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"
#if defined(UMPIRE_ENABLE_CONST)
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"
#endif
#endif

#if defined(UMPIRE_ENABLE_SYCL)
#include <CL/sycl.hpp>

#include "umpire/resource/SyclDeviceResourceFactory.hpp"
#include "umpire/resource/SyclPinnedMemoryResourceFactory.hpp"
#include "umpire/resource/SyclUnifiedMemoryResourceFactory.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include <omp.h>

#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"
#endif

#include <iterator>
#include <memory>
#include <sstream>

#include "umpire/Umpire.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/ZeroByteHandler.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/io.hpp"
#include "umpire/util/make_unique.hpp"
#include "umpire/util/wrap_allocator.hpp"

static const char* s_null_resource_name{"__umpire_internal_null"};
static const char* s_zero_byte_pool_name{"__umpire_internal_0_byte_pool"};

namespace umpire {

ResourceManager& ResourceManager::getInstance()
{
  static ResourceManager resource_manager;

  UMPIRE_LOG(Debug, "() returning " << &resource_manager);
  return resource_manager;
}

ResourceManager::ResourceManager()
    : m_allocations(),
      m_allocators(),
      m_allocators_by_id(),
      m_allocators_by_name(),
      m_memory_resources(),
      m_default_allocator(nullptr),
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

  registry.registerMemoryResource(
      util::make_unique<resource::HostResourceFactory>());
  m_resource_names.push_back("HOST");

  registry.registerMemoryResource(
      util::make_unique<resource::NullMemoryResourceFactory>());

#if defined(UMPIRE_ENABLE_BOOST_RESOURCE)
  registry.registerMemoryResource(
      util::make_unique<resource::BoostMemoryResourceFactory>());
  m_resource_names.push_back("SHARED");
#endif

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
  registry.registerMemoryResource(
      util::make_unique<resource::FileMemoryResourceFactory>());
  m_resource_names.push_back("FILE");
#endif

#if defined(UMPIRE_ENABLE_CUDA)
  {
    int device_count{0};
    auto error = ::cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
      UMPIRE_ERROR("Umpire compiled with CUDA support but no GPUs detected!");
    }

    registry.registerMemoryResource(
        util::make_unique<resource::CudaDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    int current_device;
    cudaGetDevice(&current_device);
    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
      cudaSetDevice(device);
      for (int other_device = 0; other_device < device_count; other_device++) {
        if (device != other_device) {
          cudaDeviceEnablePeerAccess(other_device, 0);
        }
      }
    }
    cudaSetDevice(current_device);

    registry.registerMemoryResource(
        util::make_unique<resource::CudaUnifiedMemoryResourceFactory>());
    m_resource_names.push_back("UM");

    registry.registerMemoryResource(
        util::make_unique<resource::CudaPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");

#if defined(UMPIRE_ENABLE_CONST)
    registry.registerMemoryResource(
        util::make_unique<resource::CudaConstantMemoryResourceFactory>());
    m_resource_names.push_back("DEVICE_CONST");
#endif
  }
#endif

#if defined(UMPIRE_ENABLE_HIP)
  {
    int device_count{0};
    auto error = ::hipGetDeviceCount(&device_count);
    if (error != hipSuccess) {
      UMPIRE_ERROR("Umpire compiled with HIP support but no GPUs detected!");
    }

    registry.registerMemoryResource(
        util::make_unique<resource::HipDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    int current_device;
    hipGetDevice(&current_device);
    for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
      hipSetDevice(device);
      for (int other_device = 0; other_device < device_count; other_device++) {
        if (device != other_device) {
          hipDeviceEnablePeerAccess(other_device, 0);
        }
      }
    }
    hipSetDevice(current_device);

    registry.registerMemoryResource(
        util::make_unique<resource::HipPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");

#if defined(UMPIRE_ENABLE_CONST)
    registry.registerMemoryResource(
        util::make_unique<resource::HipConstantMemoryResourceFactory>());
    m_resource_names.push_back("DEVICE_CONST");
#endif
  }
#endif

#if defined(UMPIRE_ENABLE_SYCL)
  {
    int device_count{0};
    auto platforms = cl::sycl::platform::get_platforms();
    for (auto& platform : platforms) {
      auto devices = platform.get_devices();
      for (auto& device : devices) {
        const std::string deviceName =
            device.get_info<cl::sycl::info::device::name>();
        if (device.is_gpu() &&
            (deviceName.find("Intel(R) Gen9 HD Graphics NEO") !=
            std::string::npos))
          device_count++;
      }
    }

    if (device_count == 0) {
      UMPIRE_ERROR("Umpire compiled with SYCL support but no GPUs detected!");
    }

    registry.registerMemoryResource(
        util::make_unique<resource::SyclDeviceResourceFactory>());
    m_resource_names.push_back("DEVICE");

    registry.registerMemoryResource(
        util::make_unique<resource::SyclUnifiedMemoryResourceFactory>());
    m_resource_names.push_back("UM");

    registry.registerMemoryResource(
        util::make_unique<resource::SyclPinnedMemoryResourceFactory>());
    m_resource_names.push_back("PINNED");
  }
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  int device_count{device_count = omp_get_num_devices()};
  for (int device = 0; device < device_count; ++device) {
      std::string name{"DEVICE::" + std::to_string(device)};
      m_resource_names.push_back(name);
  }

  registry.registerMemoryResource(
      util::make_unique<resource::OpenMPTargetResourceFactory>());
  m_resource_names.push_back("DEVICE");
#endif

  initialize();

  UMPIRE_LOG(Debug, "() leaving");
}

ResourceManager::~ResourceManager()
{
  for (auto&& allocator : m_allocators) {
    if (allocator->getCurrentSize() != 0) {
      std::stringstream ss;

      umpire::print_allocator_records(Allocator{allocator.get()}, ss);

      UMPIRE_LOG(Error, allocator->getName() << " Allocator still has "
                                             << allocator->getCurrentSize()
                                             << " bytes allocated" << std::endl
                                             << ss.str() << std::endl);
    }

    allocator.reset();
  }
}

void ResourceManager::initialize()
{
  UMPIRE_LOG(Debug, "() entering");

  UMPIRE_LOG(Debug, "Umpire v" << UMPIRE_VERSION_MAJOR << "."
                               << UMPIRE_VERSION_MINOR << "."
                               << UMPIRE_VERSION_PATCH << "."
                               << UMPIRE_VERSION_RC);

  UMPIRE_REPLAY(R"( "event": "version", "payload": { "major": )"
                << UMPIRE_VERSION_MAJOR << R"(, "minor": )"
                << UMPIRE_VERSION_MINOR << R"(, "patch": )"
                << UMPIRE_VERSION_PATCH << R"(, "rc": ")" << UMPIRE_VERSION_RC
                << R"(" })");

  resource::MemoryResourceRegistry& registry{
      resource::MemoryResourceRegistry::getInstance()};

  {
    std::unique_ptr<strategy::AllocationStrategy> allocator{
        // util::wrap_allocator<strategy::AllocationTracker>(
        registry.makeMemoryResource(s_null_resource_name, getNextId())};

    int id{allocator->getId()};
    m_allocators_by_name[s_null_resource_name] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }

  {
    std::unique_ptr<strategy::AllocationStrategy> allocator{
        new strategy::FixedPool{
            s_zero_byte_pool_name, getNextId(),
            Allocator{m_allocators_by_name[s_null_resource_name]}, 1}};

    int id{allocator->getId()};
    m_allocators_by_name[s_zero_byte_pool_name] = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));
  }

  UMPIRE_LOG(Debug, "() leaving");
}

Allocator ResourceManager::makeResource(const std::string& name)
{
  resource::MemoryResourceRegistry& registry{
      resource::MemoryResourceRegistry::getInstance()};

  return makeResource(name, registry.getDefaultTraitsForResource(name));
}

Allocator ResourceManager::makeResource(const std::string& name, MemoryResourceTraits traits)
{
  if (m_allocators_by_name.find(name) != m_allocators_by_name.end()) {
    UMPIRE_ERROR("Allocator " << name << " already exists, and cannot be re-created.");
  } else if ((name.find("DEVICE::") == std::string::npos) &&
          (name.find("SHARED::") == std::string::npos)) {
    auto resource_name = std::find(m_resource_names.begin(), m_resource_names.end(), name);
    if (resource_name == std::end(m_resource_names) ) {
        UMPIRE_ERROR("Resource " << name << " is unknown. Available resources: " 
                     << getResourceInformation());
    }
  }

  resource::MemoryResourceRegistry& registry{
      resource::MemoryResourceRegistry::getInstance()};

  if (name.find("DEVICE") != std::string::npos) {
    traits.id = resource::resource_to_device_id(name);
  }

  std::unique_ptr<strategy::AllocationStrategy> allocator{
      util::wrap_allocator<strategy::AllocationTracker,
                            strategy::ZeroByteHandler>(
          registry.makeMemoryResource(name, getNextId(), traits))};

    UMPIRE_REPLAY(
        R"( "event": "makeMemoryResource", "payload": { "name": ")" << name << R"(" })"
        << R"(, "result": ")" << allocator.get() << R"(")");

    int id{allocator->getId()};
    m_allocators_by_name[name] = allocator.get();
    if (name == "DEVICE") {
      m_allocators_by_name["DEVICE::0"] = allocator.get();
    }
    if (name.find("::0") != std::string::npos) {
      std::string base_name{name.substr(0, name.find("::") - 1)};
      m_allocators_by_name[base_name] = allocator.get();
    }
    if (name.find("::") == std::string::npos) {
      m_memory_resources[resource::string_to_resource(name)] = allocator.get();
    }
    m_default_allocator = allocator.get();
    m_allocators_by_id[id] = allocator.get();
    m_allocators.emplace_front(std::move(allocator));

    return Allocator{m_allocators_by_name[name]};
}

strategy::AllocationStrategy* ResourceManager::getAllocationStrategy(
    const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  auto allocator = m_allocators_by_name.find(name);
  if (allocator == m_allocators_by_name.end()) {
    auto resource_name = std::find(m_resource_names.begin(), m_resource_names.end(), name);
    if (resource_name != std::end(m_resource_names) ) {
      makeResource(name);
    } else {
      UMPIRE_ERROR("Allocator \"" << name
                                  << "\" not found. Available allocators: "
                                  << getAllocatorInformation());

    }
  }

  return m_allocators_by_name[name];
}

Allocator ResourceManager::getAllocator(const std::string& name)
{
  UMPIRE_LOG(Debug, "(\"" << name << "\")");
  return Allocator(getAllocationStrategy(name));
}

Allocator ResourceManager::getAllocator(const char* name)
{
  return getAllocator(std::string{name});
}

Allocator ResourceManager::getAllocator(
    resource::MemoryResourceType resource_type)
{
  UMPIRE_LOG(Debug, "(\"" << static_cast<std::size_t>(resource_type) << "\")");

  auto allocator = m_memory_resources.find(resource_type);
  if (allocator == m_memory_resources.end()) {
    return getAllocator(resource::resource_to_string(resource_type));
  } else {
    return Allocator(m_memory_resources[resource_type]);
  }
}

Allocator ResourceManager::getAllocator(int id)
{
  UMPIRE_LOG(Debug, "(\"" << id << "\")");

  if (id == umpire::invalid_allocator_id) {
    UMPIRE_ERROR("Passed umpire::invalid_allocator_id");
  }

  auto allocator = m_allocators_by_id.find(id);
  if (allocator == m_allocators_by_id.end()) {
    UMPIRE_ERROR("Allocator \"" << id << "\" not found. Available allocators: "
                                << getAllocatorInformation());
  }

  return Allocator(m_allocators_by_id[id]);
}

Allocator ResourceManager::getDefaultAllocator()
{
  UMPIRE_LOG(Debug, "");

  if (!m_default_allocator) {
    return getAllocator("HOST");
    UMPIRE_ERROR("The default Allocator is not defined");
  }

  return Allocator(m_default_allocator);
}

void ResourceManager::setDefaultAllocator(Allocator allocator) noexcept
{
  UMPIRE_LOG(Debug, "(\"" << allocator.getName() << "\")");

  UMPIRE_REPLAY(
      R"( "event": "setDefaultAllocator", "payload": { "allocator_ref": ")"
      << allocator.getAllocationStrategy() << R"(" })");

  m_default_allocator = allocator.getAllocationStrategy();
}

void ResourceManager::registerAllocator(const std::string& name,
                                        Allocator allocator)
{
  if (isAllocator(name)) {
    UMPIRE_ERROR("Allocator " << name << " is already registered.");
  }

  m_allocators_by_name[name] = allocator.getAllocationStrategy();
}

Allocator ResourceManager::getAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return Allocator(findAllocatorForPointer(ptr));
}

bool ResourceManager::isAllocator(const std::string& name) noexcept
{
  return (m_allocators_by_name.find(name) != m_allocators_by_name.end());
}

bool ResourceManager::hasAllocator(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");

  return m_allocations.contains(ptr);
}

void ResourceManager::registerAllocation(void* ptr,
                                         util::AllocationRecord record)
{
  if (!ptr) {
    UMPIRE_ERROR("Cannot register nullptr!");
  }

  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", size=" << record.size << ", strategy="
                            << record.strategy << ") with " << this);

  UMPIRE_RECORD_BACKTRACE(record);

  m_allocations.insert(ptr, record);
}

util::AllocationRecord ResourceManager::deregisterAllocation(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  return m_allocations.remove(ptr);
}

const util::AllocationRecord* ResourceManager::findAllocationRecord(
    void* ptr) const
{
  auto alloc_record = m_allocations.find(ptr);

  if (!alloc_record->strategy) {
    UMPIRE_ERROR("Cannot find allocator for " << ptr);
  }

  UMPIRE_LOG(Debug, "(Returning allocation record for ptr = " << ptr << ")");

  return alloc_record;
}

bool ResourceManager::isAllocatorRegistered(const std::string& name)
{
  return (m_allocators_by_name.find(name) != m_allocators_by_name.end());
}

void ResourceManager::copy(void* dst_ptr, void* src_ptr, std::size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", dst_ptr=" << dst_ptr
                                << ", size=" << size << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc_record = m_allocations.find(src_ptr);
  std::ptrdiff_t src_offset =
      static_cast<char*>(src_ptr) - static_cast<char*>(src_alloc_record->ptr);
  std::size_t src_size = src_alloc_record->size - src_offset;

  auto dst_alloc_record = m_allocations.find(dst_ptr);
  std::ptrdiff_t dst_offset =
      static_cast<char*>(dst_ptr) - static_cast<char*>(dst_alloc_record->ptr);
  std::size_t dst_size = dst_alloc_record->size - dst_offset;

  if (size == 0) {
    size = src_size;
  }

  UMPIRE_REPLAY(
      R"( "event": "copy", "payload": {)"
      << R"( "src": ")" << src_ptr << R"(")"
      << R"(, "src_offset": )" << src_offset << R"(, "dest": ")" << dst_ptr
      << R"(")"
      << R"(, "dst_offset": )" << dst_offset << R"(, "size": )" << size
      << R"(, "src_allocator_ref": ")" << src_alloc_record->strategy << R"(")"
      << R"(, "dst_allocator_ref": ")" << dst_alloc_record->strategy << R"(")"
      << R"( } )");

  if (size > dst_size) {
    UMPIRE_ERROR("Not enough resource in destination for copy: "
                 << size << " -> " << dst_size);
  }

  auto op = op_registry.find("COPY", src_alloc_record->strategy,
                             dst_alloc_record->strategy);

  op->transform(src_ptr, &dst_ptr, src_alloc_record, dst_alloc_record, size);
}

camp::resources::Event ResourceManager::copy(void* dst_ptr, void* src_ptr,
                                             camp::resources::Resource& ctx,
                                             std::size_t size)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << src_ptr << ", dst_ptr=" << dst_ptr
                                << ", size=" << size << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto src_alloc_record = m_allocations.find(src_ptr);
  std::ptrdiff_t src_offset =
      static_cast<char*>(src_ptr) - static_cast<char*>(src_alloc_record->ptr);
  std::size_t src_size = src_alloc_record->size - src_offset;

  auto dst_alloc_record = m_allocations.find(dst_ptr);
  std::ptrdiff_t dst_offset =
      static_cast<char*>(dst_ptr) - static_cast<char*>(dst_alloc_record->ptr);
  std::size_t dst_size = dst_alloc_record->size - dst_offset;

  if (size == 0) {
    size = src_size;
  }

  if (size > dst_size) {
    UMPIRE_ERROR("Not enough resource in destination for copy: "
                 << size << " -> " << dst_size);
  }

  auto op = op_registry.find("COPY", src_alloc_record->strategy,
                             dst_alloc_record->strategy);

  return op->transform_async(src_ptr, &dst_ptr, src_alloc_record,
                             dst_alloc_record, size, ctx);
}

void ResourceManager::memset(void* ptr, int value, std::size_t length)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ", value=" << value
                            << ", length=" << length << ")");

  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  auto alloc_record = m_allocations.find(ptr);

  std::ptrdiff_t offset =
      static_cast<char*>(ptr) - static_cast<char*>(alloc_record->ptr);
  std::size_t size = alloc_record->size - offset;

  if (length == 0) {
    length = size;
  }

  UMPIRE_REPLAY(R"( "event": "memset", "payload": { )"
                << R"( "ptr": ")" << ptr << R"(")"
                << R"(, "value": )" << value << R"(, "size": )" << size
                << R"(, "allocator_ref": ")" << alloc_record->strategy << R"(")"
                << R"( })");

  if (length > size) {
    UMPIRE_ERROR("Cannot memset over the end of allocation: "
                 << length << " -> " << size);
  }

  auto op = op_registry.find("MEMSET", alloc_record->strategy,
                             alloc_record->strategy);

  op->apply(ptr, alloc_record, value, length);
}

void* ResourceManager::reallocate(void* current_ptr, std::size_t new_size)
{
  strategy::AllocationStrategy* strategy;

  UMPIRE_REPLAY(R"( "event": "reallocate", "payload": {)"
                << R"( "ptr": ")" << current_ptr << R"(")"
                << R"(, "size": )" << new_size << R"( } )");

  if (current_ptr != nullptr) {
    auto alloc_record = m_allocations.find(current_ptr);
    strategy = alloc_record->strategy;
  } else {
    strategy = m_default_allocator;
  }

  void* new_ptr{reallocate_impl(current_ptr, new_size, Allocator(strategy))};

  UMPIRE_REPLAY(R"( "event": "reallocate", "payload": {)"
                << R"( "ptr": ")" << current_ptr << R"(")"
                << R"(, "size": )" << new_size << R"( })"
                << R"(, "result": { "memory_ptr": ")" << new_ptr << R"(" } )");

  return new_ptr;
}

void* ResourceManager::reallocate(void* current_ptr, std::size_t new_size,
                                  Allocator alloc)
{
  UMPIRE_REPLAY(R"( "event": "reallocate_ex", "payload": {)"
                << R"( "ptr": ")" << current_ptr << R"(")"
                << R"(, "size": )" << new_size << R"(, "allocator_ref": ")"
                << alloc.getAllocationStrategy() << R"(" } )");

  void* new_ptr{reallocate_impl(current_ptr, new_size, alloc)};

  UMPIRE_REPLAY(R"( "event": "reallocate_ex", "payload": {)"
                << R"( "ptr": ")" << current_ptr << R"(")"
                << R"(, "size": )" << new_size << R"(, "allocator_ref": ")"
                << alloc.getAllocationStrategy() << R"(" } )"
                << R"(, "result": { "memory_ptr": ")" << new_ptr << R"(" } )");

  return new_ptr;
}

void* ResourceManager::reallocate_impl(void* current_ptr, std::size_t new_size,
                                       Allocator allocator)
{
  UMPIRE_LOG(Debug, "(current_ptr=" << current_ptr << ", new_size=" << new_size
                                    << ", with Allocator "
                                    << allocator.getName() << ")");

  void* new_ptr;

  //
  // If this is a brand new allocation, no reallocation necessary, just allocate
  //
  if (current_ptr == nullptr) {
    new_ptr = allocator.allocate(new_size);
  } else {
    auto alloc_record = m_allocations.find(current_ptr);
    auto alloc = Allocator(alloc_record->strategy);

    if (alloc_record->strategy != allocator.getAllocationStrategy()) {
      UMPIRE_ERROR("Cannot reallocate "
                   << current_ptr << " from: " << alloc.getName()
                   << " with Allocator " << allocator.getName());
    }

    //
    // Special case 0-byte size here
    //
    if (new_size == 0) {
      alloc.deallocate(current_ptr);
      new_ptr = alloc.allocate(new_size);
    } else {
      auto& op_registry = op::MemoryOperationRegistry::getInstance();

      if (current_ptr != alloc_record->ptr) {
        UMPIRE_ERROR("Cannot reallocate an offset ptr (ptr="
                     << current_ptr << ", base=" << alloc_record->ptr);
      }

      std::shared_ptr<umpire::op::MemoryOperation> op;
      if (alloc_record->strategy->getPlatform() == Platform::host &&
          getAllocator("HOST").getId() != alloc_record->strategy->getId()) {
        op = op_registry.find(
            "REALLOCATE",
            std::make_pair(Platform::undefined, Platform::undefined));
      } else {
        op = op_registry.find("REALLOCATE", alloc_record->strategy,
                              alloc_record->strategy);
      }

      op->transform(current_ptr, &new_ptr, alloc_record, alloc_record,
                    new_size);
    }
  }

  return new_ptr;
}

void* ResourceManager::move(void* ptr, Allocator allocator)
{
  UMPIRE_LOG(Debug, "(src_ptr=" << ptr << ", allocator=" << allocator.getName()
                                << ")");

  UMPIRE_REPLAY(R"( "event": "move", "payload": {")"
                << R"( "ptr": ")" << ptr << R"(")"
                << R"(, "allocator_ref": ")"
                << allocator.getAllocationStrategy() << R"(" })");

  auto alloc_record = m_allocations.find(ptr);

  // short-circuit if ptr was allocated by 'allocator'
  if (alloc_record->strategy == allocator.getAllocationStrategy()) {
    return ptr;
  }

#if defined(UMPIRE_ENABLE_NUMA)
  {
    auto base_strategy =
        util::unwrap_allocator<strategy::AllocationStrategy>(allocator);

    // If found, use op::NumaMoveOperation to move in-place (same address
    // returned)
    if (dynamic_cast<strategy::NumaPolicy*>(base_strategy)) {
      auto& op_registry = op::MemoryOperationRegistry::getInstance();

      auto src_alloc_record = m_allocations.find(ptr);

      const std::size_t size{src_alloc_record->size};
      util::AllocationRecord dst_alloc_record{
          nullptr, size, allocator.getAllocationStrategy()};

      if (size > 0) {
        auto op = op_registry.find("MOVE", src_alloc_record->strategy,
                                   dst_alloc_record.strategy);
        void* ret{nullptr};
        op->transform(ptr, &ret, src_alloc_record, &dst_alloc_record, size);
        UMPIRE_ASSERT(ret == ptr);
      }

      UMPIRE_REPLAY(R"( "event": "move", "payload": {)"
                    << R"( "ptr": ")" << ptr << R"(")"
                    << R"(, "allocator": ")"
                    << allocator.getAllocationStrategy() << R"(" })"
                    << R"(, "result": { "ptr": ")" << ptr << R"(" })");
      return ptr;
    }
  }
#endif

  if (ptr != alloc_record->ptr) {
    UMPIRE_ERROR("Cannot move an offset ptr (ptr=" << ptr << ", base="
                                                   << alloc_record->ptr);
  }

  void* dst_ptr{allocator.allocate(alloc_record->size)};
  copy(dst_ptr, ptr);

  UMPIRE_REPLAY(R"( "event": "move", "payload": {)"
                << R"( "ptr": ")" << ptr << R"(")"
                << R"(, "allocator": ")" << allocator.getAllocationStrategy()
                << R"(" })"
                << R"(, "result": { "ptr": ")" << dst_ptr << R"(" })");

  deallocate(ptr);

  return dst_ptr;
}

void ResourceManager::deallocate(void* ptr)
{
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ")");
  auto allocator = findAllocatorForPointer(ptr);

  allocator->deallocate(ptr);
}

std::size_t ResourceManager::getSize(void* ptr) const
{
  auto record = m_allocations.find(ptr);
  UMPIRE_LOG(Debug, "(ptr=" << ptr << ") returning " << record->size);
  return record->size;
}

strategy::AllocationStrategy* ResourceManager::findAllocatorForId(int id)
{
  auto allocator_i = m_allocators_by_id.find(id);

  if (allocator_i == m_allocators_by_id.end()) {
    UMPIRE_ERROR("Cannot find allocator for ID " << id);
  }

  UMPIRE_LOG(Debug, "(id=" << id << ") returning " << allocator_i->second);
  return allocator_i->second;
}

strategy::AllocationStrategy* ResourceManager::findAllocatorForPointer(
    void* ptr)
{
  auto allocation_record = m_allocations.find(ptr);

  if (!allocation_record->strategy) {
    UMPIRE_ERROR("Cannot find allocator " << ptr);
  }

  UMPIRE_LOG(Debug,
             "(ptr=" << ptr << ") returning " << allocation_record->strategy);
  return allocation_record->strategy;
}

std::vector<std::string> ResourceManager::getAllocatorNames() const noexcept
{
  std::vector<std::string> names;
  for (auto it = m_allocators_by_name.begin(); it != m_allocators_by_name.end();
       ++it) {
    names.push_back(it->first);
  }

  UMPIRE_LOG(Debug, "() returning " << names.size() << " allocators");
  return names;
}

std::vector<int> ResourceManager::getAllocatorIds() const noexcept
{
  std::vector<int> ids;
  for (auto& it : m_allocators_by_id) {
    ids.push_back(it.first);
  }

  return ids;
}

int ResourceManager::getNextId() noexcept
{
  return m_id++;
}

std::string ResourceManager::getAllocatorInformation() const noexcept
{
  std::ostringstream info;

  for (auto& it : m_allocators_by_name) {
    info << *it.second << " ";
  }

  return info.str();
}

std::string ResourceManager::getResourceInformation() const noexcept
{
  std::ostringstream info;

  for (const auto& it : m_resource_names) {
    info << it << " ";
  }

  return info.str();
}

strategy::AllocationStrategy* ResourceManager::getZeroByteAllocator()
{
  return m_allocators_by_name[s_zero_byte_pool_name];
}

std::shared_ptr<op::MemoryOperation> ResourceManager::getOperation(
    const std::string& operation_name, Allocator src_allocator,
    Allocator dst_allocator)
{
  auto& op_registry = op::MemoryOperationRegistry::getInstance();

  return op_registry.find(operation_name, src_allocator.getAllocationStrategy(),
                          dst_allocator.getAllocationStrategy());
}

int ResourceManager::getNumDevices() const
{
  int device_count{0};
#if defined(UMPIRE_ENABLE_CUDA)
  ::cudaGetDeviceCount(&device_count);
#elif defined(UMPIRE_ENABLE_HIP)
  hipGetDeviceCount(&device_count);
#elif defined(UMPIRE_ENABLE_SYCL)
  auto platforms = cl::sycl::platform::get_platforms();
  for (auto& platform : platforms) {
    auto devices = platform.get_devices();
    for (auto& device : devices) {
      const std::string deviceName =
          device.get_info<cl::sycl::info::device::name>();
      if (device.is_gpu() &&
          (deviceName.find("Intel(R) Gen9 HD Graphics NEO") !=
           std::string::npos))
        device_count++;
    }
  }
#endif
  return device_count;
}

} // end of namespace umpire

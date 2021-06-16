//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Umpire.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

#include "umpire/config.hpp"
#include "umpire/resource/MemoryResource.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/util/wrap_allocator.hpp"

#if !defined(_MSC_VER)
#include <unistd.h>
#endif
#include <fstream>
#include <sstream>

UMPIRE_EXPORT volatile int umpire_ver_6_found = 0;

namespace umpire {

void print_allocator_records(Allocator allocator, std::ostream& os)
{
  std::stringstream ss;
  auto& rm = umpire::ResourceManager::getInstance();

  auto strategy = allocator.getAllocationStrategy();

  rm.m_allocations.print(
      [strategy](const util::AllocationRecord& rec) {
        return rec.strategy == strategy;
      },
      ss);

  if (!ss.str().empty()) {
    os << "Allocations for " << allocator.getName()
       << " allocator:" << std::endl
       << ss.str() << std::endl;
  }
}

std::vector<util::AllocationRecord> get_allocator_records(Allocator allocator)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto strategy = allocator.getAllocationStrategy();

  std::vector<util::AllocationRecord> recs;
  std::copy_if(rm.m_allocations.begin(), rm.m_allocations.end(),
               std::back_inserter(recs),
               [strategy](const util::AllocationRecord& rec) {
                 return rec.strategy == strategy;
               });

  return recs;
}

bool pointer_overlaps(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) && ((left + left_record->size) > right) &&
            ((right + right_record->size) > (left + left_record->size)));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_overlaps");
    throw;
  }
}

bool pointer_contains(void* left_ptr, void* right_ptr)
{
  auto& rm = umpire::ResourceManager::getInstance();

  try {
    auto left_record = rm.findAllocationRecord(left_ptr);
    auto right_record = rm.findAllocationRecord(right_ptr);

    char* left{reinterpret_cast<char*>(left_record->ptr)};
    char* right{reinterpret_cast<char*>(right_record->ptr)};

    return ((right >= left) && (left + left_record->size > right) &&
            (right + right_record->size <= left + left_record->size));
  } catch (umpire::util::Exception&) {
    UMPIRE_LOG(Error, "Unknown pointer in pointer_contains");
    throw;
  }
}

bool is_accessible(Platform p, Allocator a)
{
  //get base (parent) resource
  umpire::strategy::AllocationStrategy* root = a.getAllocationStrategy();
  while ((root->getParent() != nullptr)) {
    root = root->getParent();
  }

  //unwrap the base MemoryResource and return whether or not it's accessible
  umpire::resource::MemoryResource* resource =
              util::unwrap_allocation_strategy<umpire::resource::MemoryResource>(root);
  return resource->isAccessibleFrom(p);
}

std::string get_backtrace(void* ptr)
{
#if defined(UMPIRE_ENABLE_BACKTRACE)
  auto& rm = umpire::ResourceManager::getInstance();
  auto record = rm.findAllocationRecord(ptr);
  return umpire::util::backtracer<>::print(record->allocation_backtrace);
#else
  UMPIRE_USE_VAR(ptr);
  return "[Umpire: UMPIRE_BACKTRACE=Off]";
#endif
}

std::size_t get_process_memory_usage()
{
#if defined(_MSC_VER) || defined(__APPLE__)
  return 0;
#else
  std::size_t ignore;
  std::size_t resident;
  std::ifstream statm("/proc/self/statm");
  statm >> ignore >> resident >> ignore;
  statm.close();
  long page_size{::sysconf(_SC_PAGE_SIZE)};
  return std::size_t{resident * page_size};
#endif
}

void mark_event(const std::string& event)
{
  UMPIRE_REPLAY(R"( "event": "mark", "payload": { "event": ")" << event << R"(" })");
}

std::size_t get_device_memory_usage(int device_id)
{
#if defined(UMPIRE_ENABLE_CUDA)
  std::size_t mem_free{0};
  std::size_t mem_tot{0};

  int current_device;
  cudaGetDevice(&current_device);

  cudaSetDevice(device_id);

  cudaMemGetInfo(&mem_free, &mem_tot);

  cudaSetDevice(current_device);

  return std::size_t{mem_tot - mem_free};
#else
  UMPIRE_USE_VAR(device_id);
  return 0;
#endif
}

std::vector<util::AllocationRecord> get_leaked_allocations(Allocator allocator)
{
  return get_allocator_records(allocator);
}

umpire::MemoryResourceTraits get_default_resource_traits(const std::string& name)
{
  umpire::resource::MemoryResourceRegistry&
    registry{ umpire::resource::MemoryResourceRegistry::getInstance() };
  umpire::MemoryResourceTraits traits{ registry.getDefaultTraitsForResource(name) };
  return traits;
}

void* find_pointer_from_name(Allocator allocator, const std::string& name)
{
  void* ptr{nullptr};

#if defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  auto base_strategy =
          util::unwrap_allocator<strategy::AllocationStrategy>(allocator);

   umpire::resource::HostSharedMemoryResource* shared_resource =
      reinterpret_cast<umpire::resource::HostSharedMemoryResource*>(base_strategy);

  if (shared_resource != nullptr) {
    ptr = shared_resource->find_pointer_from_name(name);
  }
  else
#else
    UMPIRE_USE_VAR(name);
#endif // defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY)

  {
    if (ptr == nullptr) {
      UMPIRE_ERROR(allocator.getName()
        << " Allocator is not a Shared Memory Allocator");
    }
  }
  return ptr;
}

#if defined(UMPIRE_ENABLE_MPI)
MPI_Comm get_communicator_for_allocator(Allocator a, MPI_Comm comm) {
  static std::map<int, MPI_Comm> cached_communicators{};

  MPI_Comm c;
  auto scope = a.getAllocationStrategy()->getTraits().scope;
  int id = a.getId();

  auto cached_comm = cached_communicators.find(id);
  if (cached_comm != cached_communicators.end()) { 
    c = cached_comm->second;
  } else { 
    if (scope == MemoryResourceTraits::shared_scope::node) {
      MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &c);
    } else {
      c = MPI_COMM_NULL;
    }
    cached_communicators[id] = c;
  }

  return c;
}
#endif

} // end namespace umpire

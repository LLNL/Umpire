//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HostResourceFactory.hpp"

#include <memory>

#include "umpire/alloc/MallocAllocator.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/alloc/PosixMemalignAllocator.hpp"
#endif

#include "umpire/util/detect_vendor.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include "omp.h"
#endif

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/host_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool HostResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("HOST") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HostResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HostResourceFactory::create(const std::string& name, int id,
                                                                      MemoryResourceTraits traits)
{
#if defined(UMPIRE_ENABLE_NUMA)
  using HostAllocator = alloc::PosixMemalignAllocator;
#else
  using HostAllocator = alloc::MallocAllocator;
#endif

#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_NUMA)
  // Delegate to the API v2 HOST resource. Tracking=false (fast_host_memory)
  // so v1's ResourceManager::m_allocations remains the sole bookkeeping
  // system for these allocations -- see the "Tracking / double-tracking
  // hazard" discussion in v2_backed_resource.hpp. The wrapped instance is
  // given a distinct name ("<name>_v2backed") so it is neither the process
  // singleton (host_memory::get(), name "HOST") nor equal to the v1-visible
  // name; this also sidesteps the HOST-only v1<->v2 bridge in
  // src/umpire/memory.cpp (which keys off get_name() == "HOST"), even though
  // that bridge only fires from track_allocation() which Tracking=false
  // never calls.
  //
  // NUMA builds are excluded: v1's PosixMemalignAllocator has no v2
  // equivalent (v2's host_memory always uses malloc_allocator), so
  // delegating under UMPIRE_ENABLE_NUMA would silently change the
  // allocation backend; HOST stays NATIVE in that configuration.
  auto v2_memory = std::make_unique<resource::fast_host_memory>(name + "_v2backed");

  return util::make_unique<v2_backed_resource>(
      name, id, traits, Platform::host, std::move(v2_memory), [](Platform p) {
        alloc::MallocAllocator allocator;
        return allocator.isAccessible(p);
      });
#else
  return util::make_unique<DefaultMemoryResource<HostAllocator>>(Platform::host, name, id, traits);
#endif
}

MemoryResourceTraits HostResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  // int mib[2];
  // mib[0] = CTL_HW;
  // mib[1] = HW_MEMSIZE;

  // std::size_t mem_size;
  // std::size_t returnSize = sizeof(mem_size);
  // sysctl(mib, 2, &physicalMem, &returnSize, NULL, 0);

  traits.unified = false;
  traits.size = 0;

  traits.vendor = cpu_vendor_type();
  traits.kind = MemoryResourceTraits::memory_type::unknown;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::host;

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  traits.id = omp_get_initial_device();
#endif

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

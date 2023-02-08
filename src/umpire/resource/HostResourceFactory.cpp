//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HostResourceFactory.hpp"

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

  return util::make_unique<DefaultMemoryResource<HostAllocator>>(Platform::host, name, id, traits);
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

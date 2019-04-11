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
#include "umpire/resource/HostResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/MallocAllocator.hpp"
#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/alloc/PosixMemalignAllocator.hpp"
#endif

#include "umpire/util/detect_vendor.hpp"

namespace umpire {
namespace resource {

bool
HostResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

resource::MemoryResource*
HostResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
#if defined(UMPIRE_ENABLE_NUMA)
  using HostAllocator = alloc::PosixMemalignAllocator;
#else
  using HostAllocator = alloc::MallocAllocator;
#endif

  MemoryResourceTraits traits;

  // int mib[2];
  // mib[0] = CTL_HW;
  // mib[1] = HW_MEMSIZE;

  // size_t mem_size;
  // size_t returnSize = sizeof(mem_size);
  // sysctl(mib, 2, &physicalMem, &returnSize, NULL, 0);

  traits.unified = false;
  traits.size = 0;

  traits.vendor = cpu_vendor_type();
  traits.kind = MemoryResourceTraits::memory_type::UNKNOWN;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return new DefaultMemoryResource<HostAllocator>(Platform::cpu, "HOST", id, traits);
}

} // end of namespace resource
} // end of namespace umpire

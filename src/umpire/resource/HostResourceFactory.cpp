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
#include "umpire/resource/HostResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/MallocAllocator.hpp"

#include "umpire/resource/DetectVendor.hpp"

namespace umpire {
namespace resource {

bool
HostResourceFactory::isValidMemoryResourceFor(const std::string& name,
                                              const MemoryResourceTraits UMPIRE_UNUSED_ARG(traits))
  noexcept
{
  if (name.compare("HOST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
HostResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  // int mib[2];
  // mib[0] = CTL_HW;
  // mib[1] = HW_MEMSIZE;

  // size_t mem_size;
  // size_t returnSize = sizeof(mem_size);
  // sysctl(mib, 2, &physicalMem, &returnSize, NULL, 0);

  traits.unified = false;
  traits.size = 0;

  traits.vendor = CpuVendorType();
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return std::make_shared<DefaultMemoryResource<alloc::MallocAllocator> >(Platform::cpu, "HOST", id, traits);
}

} // end of namespace resource
} // end of namespace umpire

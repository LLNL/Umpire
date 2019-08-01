//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/RocmDeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/AmAllocAllocator.hpp"

#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool
RocmDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
RocmDeviceResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.vendor = MemoryResourceTraits::vendor_type::AMD;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return util::make_unique<resource::DefaultMemoryResource<alloc::AmAllocAllocator>>(Platform::rocm, "DEVICE", id, traits);
}

} // end of namespace resource
} // end of namespace umpire

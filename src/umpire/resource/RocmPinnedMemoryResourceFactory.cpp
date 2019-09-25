//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/RocmPinnedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/AmPinnedAllocator.hpp"

#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool
RocmPinnedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare(handle()) == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
RocmPinnedMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = true;
  traits.vendor = MemoryResourceTraits::vendor_type::AMD;
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::access;

  return util::make_unique<resource::DefaultMemoryResource<alloc::AmPinnedAllocator>>(Platform::rocm, "PINNED", id, traits);
}

std::string RocmPinnedMemoryResourceFactory::handle() const noexcept
{
  return "PINNED";
}

} // end of namespace resource
} // end of namespace umpire

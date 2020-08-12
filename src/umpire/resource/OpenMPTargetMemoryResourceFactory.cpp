//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"

#include <omp.h>

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool OpenMPTargetResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("DEVICE") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> OpenMPTargetResourceFactory::create(
    const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> OpenMPTargetResourceFactory::create(
    const std::string& name, int id, MemoryResourceTraits traits)
{
  return util::make_unique<DefaultMemoryResource<alloc::OpenMPTargetAllocator>>(
      Platform::omp_target, name, id, traits, Allocator{traits.id});
}

MemoryResourceTraits OpenMPTargetResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.id = omp_get_default_device();

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

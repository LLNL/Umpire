//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"

#include <memory>
#include <omp.h>

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/openmp_target_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool OpenMPTargetResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if ((name.find("CONST") == std::string::npos) && (name.find("DEVICE") != std::string::npos)) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> OpenMPTargetResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> OpenMPTargetResourceFactory::create(const std::string& name, int id,
                                                                              MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp. v2's default singleton is named "OMP_TARGET"
  // (distinct from the v1 "DEVICE" convention); constructing a named,
  // non-singleton instance here avoids any name collision regardless.
  auto v2_memory =
      std::make_unique<resource::openmp_target_memory<resource::omp_target_allocator, false>>(
          name + "_v2backed", traits.id);

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::omp_target, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::omp_target; });
#else
  return util::make_unique<DefaultMemoryResource<alloc::OpenMPTargetAllocator>>(Platform::omp_target, name, id, traits,
                                                                                Allocator{traits.id});
#endif
}

MemoryResourceTraits OpenMPTargetResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.id = omp_get_default_device();
  traits.resource = MemoryResourceTraits::resource_type::device;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

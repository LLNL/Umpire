//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaPinnedMemoryResourceFactory.hpp"

#include <memory>

#include "umpire/alloc/CudaPinnedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/cuda_pinned_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool CudaPinnedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("PINNED") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> CudaPinnedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> CudaPinnedMemoryResourceFactory::create(const std::string& name, int id,
                                                                                  MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp.
  auto v2_memory =
      std::make_unique<resource::cuda_pinned_memory<resource::cuda_pinned_allocator, false>>(name + "_v2backed");

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::cuda, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::cuda || p == Platform::host; });
#else
  return util::make_unique<resource::DefaultMemoryResource<alloc::CudaPinnedAllocator>>(Platform::cuda, name, id,
                                                                                        traits);
#endif
}

MemoryResourceTraits CudaPinnedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0; // size of system memory?

  traits.vendor = MemoryResourceTraits::vendor_type::nvidia;
  traits.kind = MemoryResourceTraits::memory_type::ddr;
  traits.used_for = MemoryResourceTraits::optimized_for::access;
  traits.resource = MemoryResourceTraits::resource_type::pinned;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

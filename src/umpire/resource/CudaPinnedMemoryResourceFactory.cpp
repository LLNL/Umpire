//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaPinnedMemoryResourceFactory.hpp"

#include "umpire/alloc/CudaPinnedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool CudaPinnedMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("PINNED") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
CudaPinnedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
CudaPinnedMemoryResourceFactory::create(const std::string& name, int id,
                                        MemoryResourceTraits traits)
{
  return util::make_unique<
      resource::DefaultMemoryResource<alloc::CudaPinnedAllocator>>(
      Platform::cuda, name, id, traits);
}

MemoryResourceTraits CudaPinnedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0; // size of system memory?

  traits.vendor = MemoryResourceTraits::vendor_type::NVIDIA;
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::access;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

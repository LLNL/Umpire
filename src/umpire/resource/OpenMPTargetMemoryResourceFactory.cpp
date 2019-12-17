//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"

#include "umpire/alloc/OpenMPTargetAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/resource/OpenMPTargetMemoryResourceFactory.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

OpenMPTargetResourceFactory::OpenMPTargetResourceFactory(int device) :
  m_device(device)
{
}

bool
OpenMPTargetResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
OpenMPTargetResourceFactory::create(const std::string& name, int id)
{
  MemoryResourceTraits traits;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  alloc::OpenMPTargetAllocator alloc;
  alloc.device = m_device;

  return util::make_unique<DefaultMemoryResource<alloc::OpenMPTargetAllocator>>(Platform::omp, "DEVICE", id, traits, alloc);
}

} // end of namespace resource
} // end of namespace umpire

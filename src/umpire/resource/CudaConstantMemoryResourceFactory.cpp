//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"

#include "umpire/resource/CudaConstantMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool CudaConstantMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.compare("DEVICE_CONST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
CudaConstantMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
CudaConstantMemoryResourceFactory::create(const std::string& name, int id,
                                          MemoryResourceTraits traits)
{
  return util::make_unique<resource::CudaConstantMemoryResource>(name, id,
                                                                 traits);
}

MemoryResourceTraits CudaConstantMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 64 * 1024;

  traits.vendor = MemoryResourceTraits::vendor_type::NVIDIA;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;

  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

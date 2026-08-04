//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"

#include <memory>

#include "umpire/resource/CudaConstantMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/cuda_device_const_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool CudaConstantMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.compare("DEVICE_CONST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> CudaConstantMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> CudaConstantMemoryResourceFactory::create(const std::string& name, int id,
                                                                                    MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp. Note: all named cuda_device_const_memory
  // instances share the SAME fixed 64KB __constant__ buffer (see
  // include/umpire/resource/cuda_device_const_memory.hpp), matching v1's
  // single file-scope __constant__ buffer semantics exactly.
  auto v2_memory =
      std::make_unique<resource::cuda_device_const_memory<resource::cuda_device_const_allocator, false>>(
          name + "_v2backed");

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::cuda, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::cuda; });
#else
  return util::make_unique<resource::CudaConstantMemoryResource>(name, id, traits);
#endif
}

MemoryResourceTraits CudaConstantMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 64 * 1024;

  traits.vendor = MemoryResourceTraits::vendor_type::nvidia;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.resource = MemoryResourceTraits::resource_type::device_const;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire

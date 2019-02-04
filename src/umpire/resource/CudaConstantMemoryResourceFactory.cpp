//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/resource/CudaConstantMemoryResourceFactory.hpp"

#include "umpire/resource/CudaConstantMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

bool
CudaConstantMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name,
                                                            const MemoryResourceTraits UMPIRE_UNUSED_ARG(traits))
  noexcept
{
  if (name.compare("DEVICE_CONST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
CudaConstantMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 64*1024;

  traits.vendor = MemoryResourceTraits::vendor_type::NVIDIA;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;

  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return std::make_shared<resource::CudaConstantMemoryResource >("DEVICE_CONST", id, traits);
}

} // end of namespace resource
} // end of namespace umpire

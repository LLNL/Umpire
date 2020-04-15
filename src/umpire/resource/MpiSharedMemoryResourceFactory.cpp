//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/MpiSharedMemoryResourceFactory.hpp"

#include "umpire/resource/MpiSharedMemoryResource.hpp"

#include "umpire/util/make_unique.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

bool
MpiSharedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare("MPI_SHARED_MEM") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
MpiSharedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
MpiSharedMemoryResourceFactory::create(const std::string& name, int id, MemoryResourceTraits traits)
{
  return util::make_unique<MpiSharedMemoryResource>(Platform::mpi_shmem, name, id, traits);
}

MemoryResourceTraits
MpiSharedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::UNKNOWN;
  traits.kind = MemoryResourceTraits::memory_type::UNKNOWN;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  
  return traits;
}

} // end of namespace resource
} // end of namespace umpire

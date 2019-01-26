//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "umpire/resource/NumaMemoryResourceFactory.hpp"

#include "umpire/resource/NumaMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

#if defined(UMPIRE_ENABLE_NUMA)
#include <numa.h>
#endif

namespace umpire {
namespace resource {

NumaMemoryResourceFactory::NumaMemoryResourceFactory(const int numa_node_)
  : numa_node(numa_node_) {}

std::size_t
NumaMemoryResourceFactory::getNumberOfNumaNodes() {
#if defined(UMPIRE_ENABLE_NUMA)
  if (numa_available() < 0) UMPIRE_ERROR("libnuma is unusable.");
  return numa_max_possible_node() + 1;
#else
  return 0;
#endif
}

bool
NumaMemoryResourceFactory::isValidMemoryResourceFor(const std::string& UMPIRE_UNUSED_ARG(name),
                                                    const MemoryResourceTraits traits)
  noexcept
{
  return (traits.numa_node == numa_node);
}

std::shared_ptr<MemoryResource>
NumaMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;
  traits.numa_node = numa_node;

  traits.vendor = MemoryResourceTraits::vendor_type::IBM;
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return std::make_shared<resource::NumaMemoryResource >(id, traits);
}

} // end of namespace resource
} // end of namespace umpire

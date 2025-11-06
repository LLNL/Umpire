//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/allocation_metadata.hpp"

#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/backtrace.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace util {

bool supportsHeaderIntrospection(strategy::AllocationStrategy* strategy)
{
  // Get platform and traits from strategy
  Platform platform = strategy->getPlatform();
  MemoryResourceTraits traits = strategy->getTraits();

  UMPIRE_LOG(Debug, "Checking header support for " << strategy->getName() << ", platform: " << platform_to_string(platform)
                                                    << ", unified: " << traits.unified);

  // Header introspection only works with host-accessible memory
  switch (platform) {
    case Platform::host:
      // Host memory always supports headers
      return true;

#ifdef UMPIRE_ENABLE_CUDA
    case Platform::cuda:
      // CUDA unified/managed memory supports direct host access
      return traits.unified;
#endif

#ifdef UMPIRE_ENABLE_HIP
    case Platform::hip:
      // HIP managed memory supports direct host access
      return traits.unified;
#endif

#ifdef UMPIRE_ENABLE_SYCL
    case Platform::sycl:
      // SYCL USM shared memory supports direct host access
      // Check if resource type is shared (not device-only)
      return (traits.resource == MemoryResourceTraits::resource_type::shared);
#endif

    default:
      // Other platforms (omp_target, etc.) default to no header support
      return false;
  }
}

} // end of namespace util
} // end of namespace umpire

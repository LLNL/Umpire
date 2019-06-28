//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

std::function<bool(const strategy::DynamicPool&)> heuristic_percent_releasable( int percentage )
{
  if ( percentage < 0 || percentage > 100 ) {
    UMPIRE_ERROR("Invalid percentage of " << percentage 
        << ", percentage must be an integer between 0 and 100");
  }

  if ( percentage == 0 ) {
    return [=] (const strategy::DynamicPool& UMPIRE_UNUSED_ARG(pool)) {
        return false;
    };
  }
  else if ( percentage == 100 ) {
    return [=] (const strategy::DynamicPool& pool) {
        return (pool.getCurrentSize() == 0 && pool.getReleasableSize() > 0);
    };
  }

  float f = (float)((float)percentage / (float)100.0);

  return [=] (const strategy::DynamicPool& pool) {
    // Calculate threshold in bytes from the percentage
    const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
    return (pool.getReleasableSize() >= threshold);
  };
}

} // end of namespace strategy
} // end of namespace umpire

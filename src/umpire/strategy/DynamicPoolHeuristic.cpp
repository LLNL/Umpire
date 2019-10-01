//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

std::function<bool(const strategy::DynamicPoolMap&)> heuristic_percent_releasable( int percentage )
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
    return [=] (const strategy::DynamicPoolMap& pool) {
        return (pool.getCurrentSize() == 0 && pool.getReleasableSize() > 0);
    };
  }

  float f = (float)((float)percentage / (float)100.0);

  return [=] (const strategy::DynamicPoolMap& pool) {
    // Calculate threshold in bytes from the percentage
    const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
    return (pool.getReleasableSize() >= threshold);
  };
}

std::function<bool(const strategy::DynamicPoolList&)> heuristic_percent_releasable_list( int percentage )
{
  if ( percentage < 0 || percentage > 100 ) {
    UMPIRE_ERROR("Invalid percentage of " << percentage 
        << ", percentage must be an integer between 0 and 100");
  }

  if ( percentage == 0 ) {
    return [=] (const strategy::DynamicPoolList& UMPIRE_UNUSED_ARG(pool)) {
        return false;
    };
  }
  else if ( percentage == 100 ) {
    return [=] (const strategy::DynamicPoolList& pool) {
        return (pool.getCurrentSize() == 0 && pool.getReleasableSize() > 0);
    };
  }

  float f = (float)((float)percentage / (float)100.0);

  return [=] (const strategy::DynamicPoolList& pool) {
    // Calculate threshold in bytes from the percentage
    const std::size_t threshold = static_cast<std::size_t>(f * pool.getActualSize());
    return (pool.getReleasableSize() >= threshold);
  };
}

} // end of namespace strategy
} // end of namespace umpire

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
      const std::size_t threshold = f * pool.getActualSize();
      return (pool.getReleasableSize() >= threshold);
  };
}

} // end of namespace strategy
} // end of namespace umpire

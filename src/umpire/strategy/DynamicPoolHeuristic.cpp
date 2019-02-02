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
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {

std::function<bool(const strategy::DynamicPool&)> heuristic_percent_releasable( int percentage )
{
  float f = (float)((float)percentage / (float)100.0);

  return [=] (const strategy::DynamicPool& pool) {
      const long threshold = f * pool.getActualSize();
      return (pool.getReleasableSize() >= threshold);
  };
}

} // end of namespace strategy
} // end of namespace umpire

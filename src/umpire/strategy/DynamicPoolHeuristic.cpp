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

bool heuristic_90_percent_releaseable( const strategy::DynamicPool& dynamic_pool )
{
  const float percentage = 0.90;
  const long threshold = percentage * dynamic_pool.getActualSize();

  return (dynamic_pool.getReleaseableSize() > threshold);
}

bool heuristic_100_percent_releaseable( const strategy::DynamicPool& dynamic_pool )
{
  // const long threshold = dynamic_pool.getActualSize();

  return (dynamic_pool.getCurrentSize() == 0 && dynamic_pool.getReleaseableSize() > 0);
}

} // end of namespace strategy
} // end of namespace umpire

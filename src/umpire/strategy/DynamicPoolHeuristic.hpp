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
#ifndef UMPIRE_DynamicPoolHeuristic_HPP
#define UMPIRE_DynamicPoolHeuristic_HPP

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {
class DynamicPool;

  static inline bool heuristicNoop( const strategy::DynamicPool& UMPIRE_UNUSED_ARG(dynamic_pool) ) { return false; }
  
#if 0
  bool heuristicAllAllocationsAreReleaseable( const strategy::DynamicPool& dynamic_pool );
#endif

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolHeuristic_HPP

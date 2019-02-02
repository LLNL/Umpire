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

#include <functional>
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace strategy {
class DynamicPool;

  /*!
   * \brief Return true if specified percentage of pool is releasable
   *
   * When the specified percentage of the pool has been deallocated back to the
   * pool, this heuristic will return true.
   *
   * \param percentage The integer percentage of releasable memory to actual
   * memory used by the pool.
   *
   * \return True if specified percentage of memory in pool is releasable.
   */
  std::function<bool(const strategy::DynamicPool&)> heuristic_percent_releasable( int percentage );

  /*!
   * \brief Default action
   *
   * This is the default heuristic for the dynamic pool.
   *
   * \return false always
   */
  static inline bool heuristic_noop( const strategy::DynamicPool& UMPIRE_UNUSED_ARG(dynamic_pool) ) { return false; }
  
} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolHeuristic_HPP

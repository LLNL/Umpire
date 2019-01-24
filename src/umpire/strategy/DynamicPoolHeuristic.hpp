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

  /*!
   * \brief Return true if everything in pool is releaseable
   *
   * When everything has been deallocated back to the pool, this heuristic
   * will return true.
   *
   * \param dynamic_pool The dynamic pool object in question.
   *
   * \return True if all memory in pool is releaseable.
   */
  bool heuristic_all_allocations_are_releaseable( const strategy::DynamicPool& dynamic_pool );

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

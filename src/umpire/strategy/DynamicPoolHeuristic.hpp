//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_DynamicPoolHeuristic_HPP
#define UMPIRE_DynamicPoolHeuristic_HPP

#include <functional>

namespace umpire {
namespace strategy {

class DynamicPoolList;
class DynamicPoolMap;

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
  std::function<bool(const strategy::DynamicPoolList&)> heuristic_percent_releasable_list( int percentage );

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
  std::function<bool(const strategy::DynamicPoolMap&)> heuristic_percent_releasable( int percentage );

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_DynamicPoolHeuristic_HPP

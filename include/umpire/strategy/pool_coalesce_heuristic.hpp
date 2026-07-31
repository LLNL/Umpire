//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_strategy_pool_coalesce_heuristic_HPP
#define UMPIRE_strategy_pool_coalesce_heuristic_HPP

#include <cstddef>
#include <functional>

namespace umpire {
namespace strategy {

//! @brief Callable deciding when a pool should coalesce its free blocks.
//!
//! The heuristic is invoked by the pool after each deallocation and by
//! explicit coalesce() calls. A return value of zero means "do not coalesce";
//! any non-zero value is the suggested post-coalesce pool size in bytes.
//!
//! @tparam Pool The concrete pool type the heuristic inspects
template <typename Pool>
using pool_coalesce_heuristic = std::function<std::size_t(const Pool&)>;

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_strategy_pool_coalesce_heuristic_HPP

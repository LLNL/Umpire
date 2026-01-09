//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_PoolCoalesceHeuristic_HPP
#define UMPIRE_PoolCoalesceHeuristic_HPP

#include <functional>

namespace umpire {

namespace strategy {

template <typename T>
using PoolCoalesceHeuristic = std::function<std::size_t(const T&)>;

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_PoolCoalesceHeuristic_HPP

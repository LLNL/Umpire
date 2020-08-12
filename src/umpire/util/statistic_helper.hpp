//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_statistic_helper_HPP
#define UMPIRE_statistic_helper_HPP

#include "conduit/conduit.hpp"
#include "umpire/util/StatisticsDatabase.hpp"

namespace umpire {
namespace util {
namespace detail {

inline conduit::Node add_entry(conduit::Node& n)
{
  return n;
}

template <typename T, typename U>
inline conduit::Node add_entry(conduit::Node& n, T k, U v)
{
  n[k] = v;
  return n;
}

template <typename T, typename U, typename... Args>
inline conduit::Node add_entry(conduit::Node& n, T k, U v, Args... args)
{
  n[k] = v;
  return add_entry(n, args...);
}

// template <typename T, typename U>
// conduit::Node
// add_entry(conduit::Node n, T k, U v)
//{
//  n[k] = v;
//  return n;
//}

template <typename... Args>
inline void record_statistic(const std::string& name, Args&&... args)
{
  auto node = conduit::Node{};
  util::StatisticsDatabase::getDatabase()->getStatistic(name)->recordStatistic(
      add_entry(node, args...));
}

} // end of namespace detail
} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_statistic_helper_HPP

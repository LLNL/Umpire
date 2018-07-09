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
#ifndef UMPIRE_statistic_helper_HPP
#define UMPIRE_statistic_helper_HPP

#include "umpire/util/StatisticsDatabase.hpp"

#include "conduit.hpp"

namespace umpire {
namespace util {
namespace detail {

inline
conduit::Node
add_entry(conduit::Node& n)
{
  return n;
}

template <typename T, typename U>
inline
conduit::Node
add_entry(conduit::Node& n, T k, U v)
{
  n[k] = v; 
  return n;
}

template <typename T, typename U, typename... Args>
inline
conduit::Node
add_entry(conduit::Node& n, T k, U v, Args... args)
{
  n[k] = v; 
  return add_entry(n, args...);
}

//template <typename T, typename U>
//conduit::Node
//add_entry(conduit::Node n, T k, U v)
//{
//  n[k] = v; 
//  return n;
//}


template<typename... Args>
inline
void
record_statistic(const std::string& name, Args&&... args) {
  auto node = conduit::Node{};
  util::StatisticsDatabase::getDatabase()->getStatistic(name)->recordStatistic(add_entry(node, args...));
}

} // end of namespace detail
} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_statistic_helper_HPP

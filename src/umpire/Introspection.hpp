//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Introspection_HPP
#define UMPIRE_Introspection_HPP

#include <string>

namespace umpire {

/*!
 * \brief Controls how much allocation metadata Umpire records for tracked allocations.
 *
 * - Low:    Track pointer->(allocator,size) only (fastest, non-invasive).
 * - Medium: Low + track allocation names (from named allocations).
 * - High:   Medium + track allocation backtraces (if enabled via UMPIRE_BACKTRACE).
 */
enum class IntrospectionLevel { Low, Medium, High };

inline std::string to_string(IntrospectionLevel level)
{
  switch (level) {
    case IntrospectionLevel::Low:
      return "low";
    case IntrospectionLevel::Medium:
      return "medium";
    case IntrospectionLevel::High:
      return "high";
  }
  return "high";
}

} // end namespace umpire

#endif // UMPIRE_Introspection_HPP

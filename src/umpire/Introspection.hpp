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
 * - Off:   Disable public introspection tracking.
 * - Basic: Track exact allocation-pointer ownership only.
 * - On:    Track full allocation metadata and backtraces (if enabled).
 */
enum class IntrospectionLevel { Off, Basic, On };

inline std::string to_string(IntrospectionLevel level)
{
  switch (level) {
    case IntrospectionLevel::Off:
      return "off";
    case IntrospectionLevel::Basic:
      return "basic";
    case IntrospectionLevel::On:
      return "on";
  }
  return "on";
}

} // end namespace umpire

#endif // UMPIRE_Introspection_HPP

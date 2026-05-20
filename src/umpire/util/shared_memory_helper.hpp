//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_shared_memory_helper_HPP
#define UMPIRE_shared_memory_helper_HPP

#include <string>

#include "umpire/config.hpp"

namespace umpire {
namespace util {

inline bool matchesSharedMemoryResource(const std::string& name, const std::string& resource_type) noexcept
{
  const std::string prefix = "SHARED::" + resource_type;

  // Check if name starts with "SHARED::" + resource_type
  if (name.find(prefix) == 0) {
    return true;
  }

  // Check if name starts with "SHARED::" or "SHARED" AND that this resource_type is the default
  if ((name.find("SHARED::") == 0) || (name == "SHARED")) {
    return std::string(umpire::default_shared_memory_resource) == resource_type;
  }

  return false;
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_shared_memory_helper_HPP

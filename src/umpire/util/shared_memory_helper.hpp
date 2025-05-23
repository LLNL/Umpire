//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_shared_memory_helper_HPP
#define UMPIRE_shared_memory_helper_HPP

#include <string>

namespace umpire {
namespace util {

inline bool matchesSharedMemoryResource(const std::string& name, const std::string& resource_type) noexcept
{
  const std::string prefix = "SHARED::" + resource_type;

  // Check if name starts with "SHARED::" + resource_type
  if (name.find(prefix) == 0) {
    return true;
  }

  // Check if name starts with "SHARED::" and this resource_type is the default
  if (name.find("SHARED::") == 0) {
#ifdef UMPIRE_DEFAULT_SHARED_MEMORY_RESOURCE
    return std::string(UMPIRE_DEFAULT_SHARED_MEMORY_RESOURCE) == resource_type;
#else
    return false;
#endif
  }

  return false;
}

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_shared_memory_helper_HPP
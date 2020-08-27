//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

#include "umpire/util/Macros.hpp"

#include <string>
#include <cstddef>

namespace umpire {
namespace resource {

struct MemoryResourceTypeHash {
  template <typename T>
  std::size_t operator()(T t) const noexcept
  {
    return static_cast<std::size_t>(t);
  }
};

enum MemoryResourceType { Host, Device, Unified, Pinned, Constant };

inline std::string resource_to_string(MemoryResourceType type)
{
  switch (type) {
  case Host:
    return "HOST";
  case Device:
    return "DEVICE";
  case Unified:
    return "UM";
  case Pinned:
    return "PINNED";
  case Constant:
    return "DEVICE_CONST";
  }
}

inline MemoryResourceType string_to_resource(const std::string& str)
{
  if (str == "HOST") return MemoryResourceType::Host;
  else if (str == "DEVICE") return MemoryResourceType::Device;
  else if (str == "UM") return MemoryResourceType::Unified;
  else if (str == "PINNED") return MemoryResourceType::Pinned;
  else if (str == "DEVICE_CONST") return MemoryResourceType::Constant;
  else UMPIRE_ERROR("");
}

} // end of namespace resource
} // end of namespace umpire

#endif

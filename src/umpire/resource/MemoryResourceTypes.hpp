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

enum MemoryResourceType { Host, Device, Unified, Pinned, Constant, File, Shared, Unknown };

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
    case File:
      return "FILE";
    case Shared:
      return "SHARED";
    default: 
      UMPIRE_ERROR("Unkown resource type: " << type);
  }
}

inline MemoryResourceType string_to_resource(const std::string& resource)
{
  if (resource == "HOST") return MemoryResourceType::Host;
  else if (resource == "DEVICE") return MemoryResourceType::Device;
  else if (resource == "UM") return MemoryResourceType::Unified;
  else if (resource == "PINNED") return MemoryResourceType::Pinned;
  else if (resource == "DEVICE_CONST") return MemoryResourceType::Constant;
  else if (resource == "FILE") return MemoryResourceType::File;
  else if (resource == "SHARED") return MemoryResourceType::Shared;
  else {
    UMPIRE_ERROR("Unkown resource name: " << resource);
  }
}

inline int resource_to_device_id(const std::string& resource) {
  int device_id{0};
  if (resource.find("::") != std::string::npos) {
    device_id = std::stoi(resource.substr(resource.find("::") + 2));
  }
  return device_id;
}

} // end of namespace resource
} // end of namespace umpire

#endif

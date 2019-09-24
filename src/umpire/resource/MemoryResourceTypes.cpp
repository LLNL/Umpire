//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/Macros.hpp"

#include "umpire/resource/MemoryResourceTypes.hpp"

namespace umpire {
namespace resource {

std::string type_to_string(MemoryResourceType type)
{
  switch (type) {
  case Host:
    return "HOST";
  case Device:
    return "DEVICE";
  case Unified:
    return "UM";
  case Pinned:
    return "DEVICE_PINNED";
  case Constant:
    return "DEVICE_CONSTANT";
  default:
    UMPIRE_ERROR("Unhandled enum value");
  }
}

MemoryResourceType string_to_type(const std::string& str)
{
  if (str == "HOST") return MemoryResourceType::Host;
  else if (str == "DEVICE") return MemoryResourceType::Device;
  else if (str == "UM") return MemoryResourceType::Unified;
  else if (str == "DEVICE_PINNED") return MemoryResourceType::Pinned;
  else if (str == "DEVICE_CONSTANT") return MemoryResourceType::Constant;
  else UMPIRE_ERROR("Could not parse string representation of MemoryResourceType");
}

} // end of namespace resource
} // end of namespace umpire

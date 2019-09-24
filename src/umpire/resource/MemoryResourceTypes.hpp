//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

#include <string>

namespace umpire {
namespace resource {

struct MemoryResourceTypeHash
{
    template <typename T>
    std::size_t operator()(T t) const noexcept
    {
        return static_cast<std::size_t>(t);
    }
};

// When this is updated, make sure to update the functions below
enum MemoryResourceType {
  Host,
  Device,
  Unified,
  Pinned,
  Constant
};

std::string type_to_string(MemoryResourceType type);
MemoryResourceType string_to_type(const std::string& string);

} // end of namespace resource
} // end of namespace umpire

#endif

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

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

} // end of namespace resource
} // end of namespace umpire

#endif

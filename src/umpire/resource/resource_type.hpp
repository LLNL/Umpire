//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#pragma once

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


enum class resource_type {
  host,
  device,
  unified,
  pinned,
  constant
};

} // end of namespace resource
} // end of namespace umpire

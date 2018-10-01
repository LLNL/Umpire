//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_MemoryResourceTypes_HPP
#define UMPIRE_MemoryResourceTypes_HPP

namespace umpire {
namespace resource {

struct MemoryResourceTypeHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};


enum MemoryResourceType {
  Host,
  Device,
  Unified,
  Pinned
  DeviceConst
};

} // end of namespace resource
} // end of namespace umpire

#endif

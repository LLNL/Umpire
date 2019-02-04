//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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

#include <string>

namespace umpire {
namespace resource {

// Update memory_resource_name when making changes here.
enum MemoryResourceType {
  Host,
  Device,
  Unified,
  Pinned,
  Constant
};

std::string type_to_string(const MemoryResourceType type);

} // end of namespace resource
} // end of namespace umpire

#endif

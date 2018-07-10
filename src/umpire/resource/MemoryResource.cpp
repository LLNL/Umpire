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
#include "umpire/resource/MemoryResource.hpp"

namespace umpire {
namespace resource {

MemoryResource::MemoryResource(const std::string& name, int id) :
  strategy::AllocationStrategy(name, id)
{
}

} // end of namespace resource
} // end of namespace umpire

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
#include "umpire/resource/IpcResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/Mpi3Allocator.hpp"

namespace umpire {
namespace resource {

bool
IpcResourceFactory::isValidMemoryResourceFor(const std::string& name)
{
  if (name.compare("IPC") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
IpcResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  return std::make_shared<DefaultMemoryResource<alloc::Mpi3Allocator> >(Platform::cpu, "IPC", id);
}

} // end of namespace resource
} // end of namespace umpire

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
#ifndef UMPIRE_ResourceManager_INL
#define UMPIRE_ResourceManager_INL

#include "umpire/ResourceManager.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {

template <typename Strategy,
         typename... Args>
Allocator ResourceManager::makeAllocator(
    const std::string& name, 
    Args&&... args)
{
  std::shared_ptr<strategy::AllocationStrategy> allocator;
  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

    if (isAllocator(name)) {
      UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
    }

    allocator = std::make_shared<Strategy>(name, getNextId(), std::forward<Args>(args)...);

    m_allocators_by_name[name] = allocator;
    m_allocators_by_id[allocator->getId()] = allocator;
    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return Allocator(allocator);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

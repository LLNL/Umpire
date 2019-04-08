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
#ifndef UMPIRE_ResourceManager_INL
#define UMPIRE_ResourceManager_INL

#include "umpire/ResourceManager.hpp"

#include <sstream>
#include <cxxabi.h>

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"
#include "umpire/strategy/AllocationTracker.hpp"

namespace umpire {

template <typename Strategy,
         bool introspection,
         typename... Args>
Allocator ResourceManager::makeAllocator(
    const std::string& name, 
    Args&&... args)
{
  strategy::AllocationStrategy* allocator;

  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

    if (isAllocator(name)) {
      UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
    }

    if (!introspection) {
      allocator = new Strategy(name, getNextId(), std::forward<Args>(args)...);

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;
    } else {
      std::stringstream base_name;
      base_name << name << "_base";

      auto base_allocator = new Strategy(base_name.str(), getNextId(), std::forward<Args>(args)...);

      allocator = new umpire::strategy::AllocationTracker(name, getNextId(), Allocator(base_allocator));

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;

    }

    UMPIRE_REPLAY("makeAllocator,"
        << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
        << "," << (introspection ? "true" : "false")
        << "," << name
        << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
        << "," << allocator
    );

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return Allocator(allocator);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

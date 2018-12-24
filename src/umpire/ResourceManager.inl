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
  std::shared_ptr<strategy::AllocationStrategy> allocator;

  try {
    UMPIRE_LOCK;

    UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

    UMPIRE_REPLAY("makeAllocator,"
        << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
        << "," << (introspection ? "true" : "false")
        << "," << name
        << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
    );

    if (isAllocator(name)) {
      UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
    }

    if (!introspection) {
      allocator = std::make_shared<Strategy>(name, getNextId(), std::forward<Args>(args)...);

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;
    } else {
      std::stringstream base_name;
      base_name << name << "_base";

      auto base_allocator = std::make_shared<Strategy>(base_name.str(), getNextId(), std::forward<Args>(args)...);

      allocator = std::make_shared<umpire::strategy::AllocationTracker>(name, getNextId(), Allocator(base_allocator));

      m_allocators_by_name[name] = allocator;
      m_allocators_by_id[allocator->getId()] = allocator;

    }

    UMPIRE_REPLAY_CONT("" << allocator << "\n");

    UMPIRE_UNLOCK;
  } catch (...) {
    UMPIRE_UNLOCK;
    throw;
  }

  return Allocator(allocator);
}

template <typename Strategy>
AllocatorBuilder ResourceManager::makeAllocator(
  const std::string& UMPIRE_UNUSED_ARG(name)) noexcept
{
  // const int id = getNextId();
  // m_allocators_by_id[id] = // ...
  // m_allocators_by_name[name] = // ...
  return AllocatorBuilder{*this};
}


} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

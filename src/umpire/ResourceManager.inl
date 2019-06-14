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

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif


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

  // Aquire lock
  std::lock_guard<std::mutex> lock(m_mutex);
  UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

#if defined(_MSC_VER)
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
  );
#else
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
  );
#endif
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

#if defined(_MSC_VER)
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      << ", \"result\": { \"allocator_ref\":\"" << allocator << "\" }"
  );
#else
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::replay::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      << ", \"result\": { \"allocator_ref\":\"" << allocator << "\" }"
  );
#endif

  return Allocator(allocator);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

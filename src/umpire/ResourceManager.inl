//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unique_ptr<strategy::AllocationStrategy> allocator;

  UMPIRE_LOG(Debug, "(name=\"" << name << "\")");

#if defined(_MSC_VER)
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      );
#else
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      );
#endif
  if (isAllocator(name)) {
    UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
  }

  if (!introspection) {
    allocator.reset(new Strategy(name, getNextId(), std::forward<Args>(args)...));
  } else {
    std::stringstream base_name;
    base_name << name << "_base";
    std::unique_ptr<strategy::AllocationStrategy> base_allocator{new Strategy(base_name.str(), getNextId(), std::forward<Args>(args)...)};
    allocator.reset(new umpire::strategy::AllocationTracker(name, getNextId(), std::move(base_allocator)));
  }

#if defined(_MSC_VER)
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      << ", \"result\": { \"allocator_ref\":\"" << allocator.get() << "\" }"
      );
#else
  UMPIRE_REPLAY("\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << abi::__cxa_demangle(typeid(Strategy).name(),nullptr,nullptr,nullptr)
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      << ", \"result\": { \"allocator_ref\":\"" << allocator.get() << "\" }"
      );
#endif

  m_allocators_by_name[name] = allocator.get();
  m_allocators_by_id[allocator->getId()] = allocator.get();
  m_allocators.emplace_front(std::move(allocator));

  return Allocator(m_allocators_by_name[name]);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

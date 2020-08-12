//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ResourceManager_INL
#define UMPIRE_ResourceManager_INL

#include <sstream>

#include "umpire/Replay.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationTracker.hpp"
#include "umpire/strategy/ZeroByteHandler.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"
#include "umpire/util/wrap_allocator.hpp"

namespace umpire {

template <typename Strategy, bool introspection, typename... Args>
Allocator ResourceManager::makeAllocator(const std::string& name,
                                         Args&&... args)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unique_ptr<strategy::AllocationStrategy> allocator;

  if (m_id + 1 == umpire::invalid_allocator_id) {
    UMPIRE_ERROR("Maximum number of concurrent allocators exceeded! "
                 << "Please email umpire-dev@llnl.gov");
  }

  UMPIRE_LOG(Debug, "(name=\"" << name << "\")");
  UMPIRE_REPLAY(
      "\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }");
  if (isAllocator(name)) {
    UMPIRE_ERROR("Allocator with name " << name << " is already registered.");
  }

  if (!introspection) {
    allocator = util::wrap_allocator<strategy::ZeroByteHandler>(
        util::make_unique<Strategy>(name, getNextId(),
                                    std::forward<Args>(args)...));

  } else {
    allocator = util::wrap_allocator<strategy::AllocationTracker,
                                     strategy::ZeroByteHandler>(
        util::make_unique<Strategy>(name, getNextId(),
                                    std::forward<Args>(args)...));
  }

  UMPIRE_REPLAY(
      "\"event\": \"makeAllocator\", \"payload\": { \"type\":\""
      << typeid(Strategy).name()
      << "\", \"with_introspection\":" << (introspection ? "true" : "false")
      << ", \"allocator_name\":\"" << name << "\""
      << ", \"args\": [ "
      << umpire::Replay::printReplayAllocator(std::forward<Args>(args)...)
      << " ] }"
      << ", \"result\": { \"allocator_ref\":\"" << allocator.get() << "\" }");

  m_allocators_by_name[name] = allocator.get();
  m_allocators_by_id[allocator->getId()] = allocator.get();
  m_allocators.emplace_front(std::move(allocator));

  return Allocator(m_allocators_by_name[name]);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

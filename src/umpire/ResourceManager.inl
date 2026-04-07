//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ResourceManager_INL
#define UMPIRE_ResourceManager_INL

#include <sstream>

#include "camp/list.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/event/event.hpp"
#include "umpire/replay/Replay.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {

template <typename Strategy, typename... Args>
Allocator ResourceManager::makeAllocator(const std::string& name, Tracking tracked, Args&&... args)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unique_ptr<strategy::AllocationStrategy> allocator;
  bool is_tracked = (tracked == Tracking::Tracked) ? true : false;

  if (m_id + 1 == umpire::invalid_allocator_id) {
    UMPIRE_ERROR(runtime_error, "Maximum number of concurrent allocators exceeded! Please email umpire-dev@llnl.gov");
  }

  UMPIRE_LOG(Debug, "(name=\"" << name << "\")");
  if (isAllocator(name)) {
    UMPIRE_ERROR(runtime_error, fmt::format("Allocator with name \"{}\" is already registered", name));
  }

  replay::json replay_args{};
  const bool replay_enabled = replay::is_enabled();
  replay::ReplayMakeAllocatorToken replay_token{};

#if defined(UMPIRE_ENABLE_MPI) && defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY) && \
    (defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP))
  constexpr bool suppress_nested_resources = std::is_same<Strategy, strategy::DeviceIpcAllocator>::value;
#else
  constexpr bool suppress_nested_resources = false;
#endif

  if (replay_enabled) {
#if defined(UMPIRE_ENABLE_MPI) && defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY) && \
    (defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP))
    if constexpr (suppress_nested_resources) {
      if constexpr (sizeof...(Args) == 0) {
        auto device_allocator = getAllocator("DEVICE");
        replay_args = replay::serialize_allocator_args<Strategy>();
        replay_args["device_allocator"] = replay::resolve_allocator_id(device_allocator);
      } else {
        replay_args = replay::serialize_allocator_args<Strategy>(std::forward<Args>(args)...);
      }
    } else
#endif
    {
      replay_args = replay::serialize_allocator_args<Strategy>(std::forward<Args>(args)...);
    }

    replay_token =
        replay::begin_make_allocator(name, is_tracked, replay::strategy_name<Strategy>(), replay_args);
  }

  replay::ScopedNestedReplaySuppression suppress_nested_replay(replay_enabled && suppress_nested_resources);

  allocator = util::make_unique<Strategy>(name, getNextId(), std::forward<Args>(args)...);
  allocator->setTracking(is_tracked);

  if (replay_enabled) {
    replay::commit_make_allocator(allocator.get(), replay_token);
  }

  umpire::event::record([&](auto& event) {
    event.name("make_allocator")
        .category(event::category::operation)
        .arg("allocator_ref", (void*)allocator.get())
        .arg("type", typeid(Strategy).name())
        .arg("introspection", is_tracked)
        .args(args...)
        .tag("allocator_name", allocator->getName())
        .tag("replay", "true");
  });

  m_allocators_by_name[name] = allocator.get();
  m_allocators_by_id[allocator->getId()] = allocator.get();
  m_allocators.emplace_front(std::move(allocator));

  return Allocator(m_allocators_by_name[name]);
}

template <typename Strategy, bool introspection, typename... Args>
Allocator ResourceManager::makeAllocator(const std::string& name, Args&&... args)
{
  Tracking tracked = introspection ? Tracking::Tracked : Tracking::Untracked;
  return makeAllocator<Strategy>(name, tracked, std::forward<Args>(args)...);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

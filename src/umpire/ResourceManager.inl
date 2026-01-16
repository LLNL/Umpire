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

  allocator = util::make_unique<Strategy>(name, getNextId(), std::forward<Args>(args)...);
  allocator->setTracking(is_tracked);

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

template <typename T>
void ResourceManager::deviceMemset(T* ptr, std::size_t n, const T& value)
{
  UMPIRE_LOG(Debug, "(ptr=" << static_cast<void*>(ptr) << ", n=" << n << ")");

  if (!ptr || n == 0) {
    return;
  }

   void* raw_ptr = static_cast<void*>(ptr);

   if (!hasAllocator(raw_ptr)) {
     UMPIRE_ERROR(resource_error,
                  "ResourceManager::deviceMemset called on pointer that is not managed by any Umpire allocator.");
   }

   auto owning_allocator = getAllocator(raw_ptr);
   auto platform = owning_allocator.getPlatform();

   if (platform != Platform::cuda && platform != Platform::hip) {
     UMPIRE_ERROR(resource_error,
                  "ResourceManager::deviceMemset is only supported for CUDA or HIP DEVICE allocations.");
   }

  device_memset(ptr, n, value);
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

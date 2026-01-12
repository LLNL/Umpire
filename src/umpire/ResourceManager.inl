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
T* ResourceManager::allocate_and_fill(std::size_t n, const T& value, Allocator allocator)
{
  const std::size_t size = n * sizeof(T);

  // Fast paths for memset-compatible patterns.
  if (value == T{}) {
    return static_cast<T*>(allocate_and_memset(size, 0, allocator));
  }

  if (std::is_integral<T>::value && value == static_cast<T>(-1)) {
    return static_cast<T*>(allocate_and_memset(size, 0xFF, allocator));
  }

  void* device_ptr = allocator.allocate(size);

  try {
    auto host_alloc = getAllocator("HOST");
    T* host_ptr = static_cast<T*>(host_alloc.allocate(size));

    try {
      for (std::size_t i = 0; i < n; ++i) {
        host_ptr[i] = value;
      }

      copy(device_ptr, host_ptr, size);
    } catch (...) {
      host_alloc.deallocate(host_ptr);
      throw;
    }

    host_alloc.deallocate(host_ptr);
  } catch (...) {
    allocator.deallocate(device_ptr);
    throw;
  }

  return static_cast<T*>(device_ptr);
}

template <typename T>
T* ResourceManager::allocate_and_fill(std::size_t n, const T& value)
{
  return allocate_and_fill<T>(n, value, getDefaultAllocator());
}

} // end of namespace umpire

#endif // UMPIRE_ResourceManager_INL

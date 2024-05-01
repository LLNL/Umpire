//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Allocator_INL
#define UMPIRE_Allocator_INL

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/event/event.hpp"
#include "umpire/event/recorder_factory.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {

inline void* Allocator::do_allocate(std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    try {
      ret = m_allocator->allocate(bytes);
    } catch (umpire::out_of_memory_error& e) {
      e.set_allocator_id(this->getId());
      e.set_requested_size(bytes);
      throw;
    }
  }

  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator);
  }

  umpire::event::record<umpire::event::allocate>(
      [&](auto& event) { event.size(bytes).ref((void*)m_allocator).ptr(ret); });

  return ret;
}

inline void* Allocator::thread_safe_allocate(std::size_t bytes)
{
  std::lock_guard<std::mutex> lock(*m_thread_safe_mutex);
  return do_allocate(bytes);
}

inline void* Allocator::thread_safe_named_allocate(const std::string& name, std::size_t bytes)
{
  std::lock_guard<std::mutex> lock(*m_thread_safe_mutex);
  return do_named_allocate(name, bytes);
}

inline void Allocator::thread_safe_deallocate(void* ptr)
{
  std::lock_guard<std::mutex> lock(*m_thread_safe_mutex);
  return do_deallocate(ptr);
}

inline void* Allocator::do_named_allocate(const std::string& name, std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate_named(name, bytes);
  }

  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator, name);
  }

  umpire::event::record<umpire::event::named_allocate>(
      [&](auto& event) { event.name(name).size(bytes).ref((void*)m_allocator).ptr(ret); });
  return ret;
}

inline void* Allocator::do_resource_allocate(camp::resources::Resource const& r, std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate_resource(r, bytes);
  }

  // TODO: track the resource?
  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator);
  }

  //umpire::event::record<umpire::event::named_allocate>(
  //    [&](auto& event) { event.name(name).size(bytes).ref((void*)m_allocator).ptr(ret); });
  return ret;
}

inline void Allocator::do_deallocate(void* ptr)
{
  umpire::event::record<umpire::event::deallocate>([&](auto& event) { event.ref((void*)m_allocator).ptr(ptr); });

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer (This behavior is intentionally allowed and ignored)");
    return;
  } else {
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      if (!deallocateNull(ptr)) {
        m_allocator->deallocate(ptr, record.size);
      }
    } else {
      if (!deallocateNull(ptr)) {
        m_allocator->deallocate(ptr);
      }
    }
  }
}

inline void Allocator::do_resource_deallocate(camp::resources::Resource const& r, void* ptr)
{
  //umpire::event::record<umpire::event::deallocate>([&](auto& event) { event.ref((void*)m_allocator).ptr(ptr); });

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer (This behavior is intentionally allowed and ignored)");
    return;
  } else {
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      if (!deallocateNull(ptr)) {
        m_allocator->deallocate_resource(r, ptr, record.size);
      }
    } else {
      if (!deallocateNull(ptr)) {
        m_allocator->deallocate_resource(r, ptr);
      }
    }
  }
}

inline void* Allocator::allocate(std::size_t bytes)
{
  return m_thread_safe ? thread_safe_allocate(bytes) : do_allocate(bytes);
}

//TODO: Create thread safe resource allocate?
inline void* Allocator::allocate(camp::resources::Resource const& r, std::size_t bytes)
{
  return m_thread_safe ? thread_safe_allocate(bytes) : do_resource_allocate(r, bytes);
}

inline void* Allocator::allocate(const std::string& name, std::size_t bytes)
{
  return m_thread_safe ? thread_safe_named_allocate(name, bytes) : do_named_allocate(name, bytes);
}

inline void Allocator::deallocate(void* ptr)
{
  m_thread_safe ? thread_safe_deallocate(ptr) : do_deallocate(ptr);
}

//TODO: Create thread safe resource deallocate?
inline void Allocator::deallocate(camp::resources::Resource const& r, void* ptr)
{
  m_thread_safe ? thread_safe_deallocate(ptr) : do_resource_deallocate(r, ptr);
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

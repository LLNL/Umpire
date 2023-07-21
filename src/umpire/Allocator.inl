//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {

inline void* Allocator::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (m_mutex != nullptr)
    m_mutex->lock();

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    try {
      ret = m_allocator->allocate(bytes);
    } catch (umpire::out_of_memory_error& e) {
      e.set_allocator_id(this->getId());
      e.set_requested_size(bytes);
      if (m_mutex != nullptr)
        m_mutex->unlock();
      throw;
    }
  }

  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator);
  }

  if (m_mutex != nullptr)
    m_mutex->unlock();

  umpire::event::record<umpire::event::allocate>(
      [&](auto& event) { event.size(bytes).ref((void*)m_allocator).ptr(ret); });

  return ret;
}

inline void* Allocator::allocate(const std::string& name, std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (m_mutex != nullptr)
    m_mutex->lock();

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate_named(name, bytes);
  }

  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator, name);
  }

  if (m_mutex != nullptr)
    m_mutex->unlock();

  umpire::event::record<umpire::event::named_allocate>(
      [&](auto& event) { event.name(name).size(bytes).ref((void*)m_allocator).ptr(ret); });
  return ret;
}

inline void Allocator::deallocate(void* ptr)
{
  umpire::event::record<umpire::event::deallocate>([&](auto& event) { event.ref((void*)m_allocator).ptr(ptr); });

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer (This behavior is intentionally allowed and ignored)");
    return;
  } else {
    if (m_mutex != nullptr)
      m_mutex->lock();
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
    if (m_mutex != nullptr)
      m_mutex->unlock();
  }
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
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

namespace umpire {

inline void* Allocator::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
    ret = allocateNull();
  } else {
    ret = m_allocator->allocate(bytes);
  }

  if (m_tracking) {
    registerAllocation(ret, bytes, m_allocator);
  }

  umpire::event::event::builder()
      .name("allocate")
      .category(event::category::operation)
      .arg("allocator_ref", (void*)m_allocator)
      .arg("size", bytes)
      .arg("pointer", ret)
      .tag("allocator_name", m_allocator->getName().c_str())
      .tag("replay", "true")
      .record();

  return ret;
}

inline void* Allocator::allocate(const std::string& name, std::size_t bytes)
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

  umpire::event::event::builder()
      .name("allocate")
      .category(event::category::operation)
      .arg("allocator_ref", (void*)m_allocator)
      .arg("size", bytes)
      .arg("pointer", ret)
      .arg("name", name)
      .tag("allocator_name", m_allocator->getName().c_str())
      .tag("replay", "true")
      .record();

  return ret;
}

inline void Allocator::deallocate(void* ptr)
{
  //#if defined(UMPIRE_ENABLE_EVENTS)
  umpire::event::event::builder()
      .name("deallocate")
      .category(event::category::operation)
      .arg("allocator_ref", (void*)m_allocator)
      .arg("pointer", ptr)
      .tag("allocator_name", m_allocator->getName().c_str())
      .tag("replay", "true")
      .record();
  //#endif

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

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

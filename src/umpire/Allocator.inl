//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Allocator_INL
#define UMPIRE_Allocator_INL

#include "umpire/config.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/Replay.hpp"

namespace umpire {

inline void*
Allocator::allocate(std::size_t bytes)
{
  void* ret = nullptr;

  umpire_ver_2_found = 0;

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  UMPIRE_REPLAY("\"event\": \"allocate\", \"payload\": { \"allocator_ref\": \"" << m_allocator << "\", \"size\": " << bytes << " }");

  ret = m_allocator->allocate(bytes);

  UMPIRE_REPLAY("\"event\": \"allocate\", \"payload\": { \"allocator_ref\": \"" << m_allocator << "\", \"size\": " << bytes << " }, \"result\": { \"memory_ptr\": \"" << ret << "\" }");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ret), "size", bytes, "event", "allocate");
  return ret;
}

inline void
Allocator::deallocate(void* ptr)
{
  UMPIRE_REPLAY("\"event\": \"deallocate\", \"payload\": { \"allocator_ref\": \"" << m_allocator << "\", \"memory_ptr\": \"" << ptr << "\" }");

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  UMPIRE_RECORD_STATISTIC(getName(), "ptr", reinterpret_cast<uintptr_t>(ptr), "size", 0x0, "event", "deallocate");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer");
    return;
  } else {
    m_allocator->deallocate(ptr);
  }
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

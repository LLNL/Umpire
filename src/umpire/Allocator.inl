//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
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
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // In header introspection mode, return nullptr for 0-byte allocations (no tracking)
    ret = nullptr;
#else
    // Use unique null pointer from pool for tracking
    ret = allocateNull();
    if (m_tracking) {
      registerAllocation(ret, bytes, m_allocator);
    }
#endif
  } else {
    try {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
      if (m_tracking) {
        // Request extra space for header
        std::size_t total_bytes = util::allocation_size(bytes);
        void* base_ptr = m_allocator->allocate(total_bytes);

        // Construct allocation header and get user pointer
        ret = util::construct_allocation<util::AllocationRecord>(
          base_ptr,
          util::allocation_data(base_ptr),
          bytes,
          m_allocator
        );

        // Update statistics with total allocation size (including header)
        m_allocator->m_current_size += total_bytes;
        m_allocator->m_allocation_count++;
        if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
          m_allocator->m_high_watermark = m_allocator->m_current_size;
        }
      } else {
        ret = m_allocator->allocate(bytes);
      }
#else
      // Original map-based path
      ret = m_allocator->allocate(bytes);
      if (m_tracking) {
        registerAllocation(ret, bytes, m_allocator);
      }
#endif
    } catch (umpire::out_of_memory_error& e) {
      e.set_allocator_id(this->getId());
      e.set_requested_size(bytes);
      throw;
    }
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

inline void* Allocator::thread_safe_resource_allocate(std::size_t bytes, camp::resources::Resource const& r)
{
  std::lock_guard<std::mutex> lock(*m_thread_safe_mutex);
  return do_resource_allocate(bytes, r);
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

inline void Allocator::thread_safe_resource_deallocate(void* ptr, camp::resources::Resource const& r)
{
  std::lock_guard<std::mutex> lock(*m_thread_safe_mutex);
  return do_resource_deallocate(ptr, r);
}

inline void* Allocator::do_named_allocate(const std::string& name, std::size_t bytes)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // In header introspection mode, return nullptr for 0-byte allocations (no tracking)
    ret = nullptr;
#else
    // Use unique null pointer from pool for tracking
    ret = allocateNull();
    if (m_tracking) {
      registerAllocation(ret, bytes, m_allocator, name);
    }
#endif
  } else {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    if (m_tracking) {
      // Request extra space for header
      std::size_t total_bytes = util::allocation_size(bytes);
      void* base_ptr = m_allocator->allocate_named(name, total_bytes);

      // Construct allocation header with name and get user pointer
      ret = util::construct_allocation<util::AllocationRecord>(
        base_ptr,
        util::allocation_data(base_ptr),
        bytes,
        m_allocator,
        name
      );

      // Update statistics with total allocation size (including header)
      m_allocator->m_current_size += total_bytes;
      m_allocator->m_allocation_count++;
      if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
        m_allocator->m_high_watermark = m_allocator->m_current_size;
      }
    } else {
      ret = m_allocator->allocate_named(name, bytes);
    }
#else
    // Original map-based path
    ret = m_allocator->allocate_named(name, bytes);
    if (m_tracking) {
      registerAllocation(ret, bytes, m_allocator, name);
    }
#endif
  }

  umpire::event::record<umpire::event::named_allocate>(
      [&](auto& event) { event.name(name).size(bytes).ref((void*)m_allocator).ptr(ret); });
  return ret;
}

inline void* Allocator::do_resource_allocate(std::size_t bytes, camp::resources::Resource const& r)
{
  void* ret = nullptr;

  UMPIRE_ASSERT(UMPIRE_VERSION_OK());

  UMPIRE_LOG(Debug, "(" << bytes << ")");

  if (0 == bytes) {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // In header introspection mode, return nullptr for 0-byte allocations (no tracking)
    ret = nullptr;
#else
    // Use unique null pointer from pool for tracking
    ret = allocateNull();
    if (m_tracking) {
      registerAllocation(ret, bytes, m_allocator);
    }
#endif
  } else {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    if (m_tracking) {
      // Request extra space for header
      std::size_t total_bytes = util::allocation_size(bytes);
      void* base_ptr = m_allocator->allocate_resource(total_bytes, r);

      // Construct allocation header and get user pointer
      ret = util::construct_allocation<util::AllocationRecord>(
        base_ptr,
        util::allocation_data(base_ptr),
        bytes,
        m_allocator
      );

      // Update statistics with total allocation size (including header)
      m_allocator->m_current_size += total_bytes;
      m_allocator->m_allocation_count++;
      if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
        m_allocator->m_high_watermark = m_allocator->m_current_size;
      }
    } else {
      ret = m_allocator->allocate_resource(bytes, r);
    }
#else
    // Original map-based path
    ret = m_allocator->allocate_resource(bytes, r);
    if (m_tracking) {
      registerAllocation(ret, bytes, m_allocator);
    }
#endif
  }

  umpire::event::record<umpire::event::allocate_resource>(
      [&](auto& event) { event.size(bytes).ref((void*)m_allocator).ptr(ret).res(camp::resources::to_string(r)); });
  return ret;
}

inline void Allocator::do_deallocate(void* ptr)
{
  umpire::event::record<umpire::event::deallocate>([&](auto& event) { event.ref((void*)m_allocator).ptr(ptr); });

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer (This behavior is intentionally allowed and ignored)");
    return;
  }

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // In header introspection mode, nullptr means 0-byte allocation, already handled above
  if (m_tracking) {
    // Destruct allocation header and get metadata + base pointer
    auto [record, base_ptr] = util::destruct_allocation<util::AllocationRecord>(ptr);

    // Validate strategy matches
    if (record.strategy != m_allocator) {
      UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
    }

    // Update statistics with total allocation size (including header)
    std::size_t total_bytes = util::allocation_size(record.size);
    m_allocator->m_current_size -= total_bytes;
    m_allocator->m_allocation_count--;

    // Deallocate with base pointer (must pass total bytes including header)
    m_allocator->deallocate(base_ptr, total_bytes);
  } else {
    m_allocator->deallocate(ptr);
  }
#else
  // In non-header mode, check if ptr is from the null pool
  if (!deallocateNull(ptr)) {
    // Regular allocation
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      m_allocator->deallocate(ptr, record.size);
    } else {
      m_allocator->deallocate(ptr);
    }
  } else {
    // Zero-byte allocation from null pool - deregister if tracking
    if (m_tracking) {
      deregisterAllocation(ptr, m_allocator);
    }
  }
#endif
}

inline void Allocator::do_resource_deallocate(void* ptr, camp::resources::Resource const& r)
{
  umpire::event::record<umpire::event::deallocate_resource>(
      [&](auto& event) { event.ref((void*)m_allocator).ptr(ptr).res(camp::resources::to_string(r)); });

  UMPIRE_LOG(Debug, "(" << ptr << ")");

  if (!ptr) {
    UMPIRE_LOG(Info, "Deallocating a null pointer (This behavior is intentionally allowed and ignored)");
    return;
  }

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // In header introspection mode, nullptr means 0-byte allocation, already handled above
  if (m_tracking) {
    // Destruct allocation header and get metadata + base pointer
    auto [record, base_ptr] = util::destruct_allocation<util::AllocationRecord>(ptr);

    // Validate strategy matches
    if (record.strategy != m_allocator) {
      UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
    }

    // Update statistics with total allocation size (including header)
    std::size_t total_bytes = util::allocation_size(record.size);
    m_allocator->m_current_size -= total_bytes;
    m_allocator->m_allocation_count--;

    // Deallocate with base pointer (must pass total bytes including header)
    m_allocator->deallocate_resource(base_ptr, r, total_bytes);
  } else {
    m_allocator->deallocate_resource(ptr, r);
  }
#else
  // In non-header mode, check if ptr is from the null pool
  if (!deallocateNull(ptr)) {
    // Regular allocation
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      m_allocator->deallocate_resource(ptr, r, record.size);
    } else {
      m_allocator->deallocate_resource(ptr, r);
    }
  } else {
    // Zero-byte allocation from null pool - deregister if tracking
    if (m_tracking) {
      deregisterAllocation(ptr, m_allocator);
    }
  }
#endif
}

inline void* Allocator::allocate(std::size_t bytes)
{
  return m_thread_safe ? thread_safe_allocate(bytes) : do_allocate(bytes);
}

inline void* Allocator::allocate(std::size_t bytes, camp::resources::Resource const& r)
{
  return m_thread_safe ? thread_safe_resource_allocate(bytes, r) : do_resource_allocate(bytes, r);
}

inline void* Allocator::allocate(const std::string& name, std::size_t bytes)
{
  return m_thread_safe ? thread_safe_named_allocate(name, bytes) : do_named_allocate(name, bytes);
}

inline void Allocator::deallocate(void* ptr)
{
  m_thread_safe ? thread_safe_deallocate(ptr) : do_deallocate(ptr);
}

inline void Allocator::deallocate(void* ptr, camp::resources::Resource const& r)
{
  m_thread_safe ? thread_safe_resource_deallocate(ptr, r) : do_resource_deallocate(ptr, r);
}

} // end of namespace umpire

#endif // UMPIRE_Allocator_INL

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
    ret = allocateNull();
  } else {
    try {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
      // Check if this allocator supports header mode
      if (m_tracking && util::supportsHeaderIntrospection(m_allocator)) {
        // Request extra space for header
        std::size_t total_bytes = util::getTotalSize(bytes);
        void* base_ptr = m_allocator->allocate(total_bytes);

        // Insert header and get user pointer
        ret = util::insertHeader(base_ptr, bytes, m_allocator);

        // Update statistics (normally done by registerAllocation)
        m_allocator->m_current_size += bytes;
        m_allocator->m_allocation_count++;
        if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
          m_allocator->m_high_watermark = m_allocator->m_current_size;
        }
      } else {
        // Device-only memory or tracking disabled - use map-based tracking or no tracking
        ret = m_allocator->allocate(bytes);
        if (m_tracking) {
          // Use fallback map for device-only allocations
          getFallbackMap().insert(ret, {ret, bytes, m_allocator});

          // Update statistics
          m_allocator->m_current_size += bytes;
          m_allocator->m_allocation_count++;
          if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
            m_allocator->m_high_watermark = m_allocator->m_current_size;
          }
        }
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
    ret = allocateNull();
  } else {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // Check if this allocator supports header mode
    if (m_tracking && util::supportsHeaderIntrospection(m_allocator)) {
      // Request extra space for header
      std::size_t total_bytes = util::getTotalSize(bytes);
      void* base_ptr = m_allocator->allocate_named(name, total_bytes);

      // Insert header with name and get user pointer
      ret = util::insertHeader(base_ptr, bytes, m_allocator, name);

      // Update statistics
      m_allocator->m_current_size += bytes;
      m_allocator->m_allocation_count++;
      if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
        m_allocator->m_high_watermark = m_allocator->m_current_size;
      }
    } else {
      // Device-only memory or tracking disabled
      ret = m_allocator->allocate_named(name, bytes);
      if (m_tracking) {
        getFallbackMap().insert(ret, {ret, bytes, m_allocator, name});

        // Update statistics
        m_allocator->m_current_size += bytes;
        m_allocator->m_allocation_count++;
        if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
          m_allocator->m_high_watermark = m_allocator->m_current_size;
        }
      }
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
    ret = allocateNull();
  } else {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // Check if this allocator supports header mode
    if (m_tracking && util::supportsHeaderIntrospection(m_allocator)) {
      // Request extra space for header
      std::size_t total_bytes = util::getTotalSize(bytes);
      void* base_ptr = m_allocator->allocate_resource(total_bytes, r);

      // Insert header and get user pointer
      ret = util::insertHeader(base_ptr, bytes, m_allocator);

      // Update statistics
      m_allocator->m_current_size += bytes;
      m_allocator->m_allocation_count++;
      if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
        m_allocator->m_high_watermark = m_allocator->m_current_size;
      }
    } else {
      // Device-only memory or tracking disabled
      ret = m_allocator->allocate_resource(bytes, r);
      if (m_tracking) {
        getFallbackMap().insert(ret, {ret, bytes, m_allocator});

        // Update statistics
        m_allocator->m_current_size += bytes;
        m_allocator->m_allocation_count++;
        if (m_allocator->m_current_size > m_allocator->m_high_watermark) {
          m_allocator->m_high_watermark = m_allocator->m_current_size;
        }
      }
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

  if (!deallocateNull(ptr)) {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    if (m_tracking && util::supportsHeaderIntrospection(m_allocator)) {
      // Header mode - remove header and get base pointer
      auto [record, base_ptr] = util::removeHeader(ptr);

      // Validate strategy matches
      if (record.strategy != m_allocator) {
        UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
      }

      // Update statistics
      m_allocator->m_current_size -= record.size;
      m_allocator->m_allocation_count--;

      // Deallocate with base pointer
      m_allocator->deallocate(base_ptr, record.size);
    } else {
      // Device-only or tracking disabled
      if (m_tracking) {
        // Use fallback map
        auto record = getFallbackMap().remove(ptr);

        // Validate strategy matches
        if (record.strategy != m_allocator) {
          // Re-register and throw error
          getFallbackMap().insert(ptr, record);
          UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
        }

        // Update statistics
        m_allocator->m_current_size -= record.size;
        m_allocator->m_allocation_count--;

        m_allocator->deallocate(ptr, record.size);
      } else {
        m_allocator->deallocate(ptr);
      }
    }
#else
    // Original map-based path
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      m_allocator->deallocate(ptr, record.size);
    } else {
      m_allocator->deallocate(ptr);
    }
#endif
  }
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

  if (!deallocateNull(ptr)) {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    if (m_tracking && util::supportsHeaderIntrospection(m_allocator)) {
      // Header mode - remove header and get base pointer
      auto [record, base_ptr] = util::removeHeader(ptr);

      // Validate strategy matches
      if (record.strategy != m_allocator) {
        UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
      }

      // Update statistics
      m_allocator->m_current_size -= record.size;
      m_allocator->m_allocation_count--;

      // Deallocate with base pointer
      m_allocator->deallocate_resource(base_ptr, r, record.size);
    } else {
      // Device-only or tracking disabled
      if (m_tracking) {
        // Use fallback map
        auto record = getFallbackMap().remove(ptr);

        // Validate strategy matches
        if (record.strategy != m_allocator) {
          // Re-register and throw error
          getFallbackMap().insert(ptr, record);
          UMPIRE_ERROR(runtime_error, fmt::format("{} was not allocated by {}", ptr, m_allocator->getName()));
        }

        // Update statistics
        m_allocator->m_current_size -= record.size;
        m_allocator->m_allocation_count--;

        m_allocator->deallocate_resource(ptr, r, record.size);
      } else {
        m_allocator->deallocate_resource(ptr, r);
      }
    }
#else
    // Original map-based path
    if (m_tracking) {
      auto record = deregisterAllocation(ptr, m_allocator);
      m_allocator->deallocate_resource(ptr, r, record.size);
    } else {
      m_allocator->deallocate_resource(ptr, r);
    }
#endif
  }
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

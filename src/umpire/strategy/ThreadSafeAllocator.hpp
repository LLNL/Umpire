//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ThreadSafeAllocator_HPP
#define UMPIRE_ThreadSafeAllocator_HPP

#include <memory>
#include <mutex>

#include "umpire/Allocator.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/strategy/detail/v1_backed_memory.hpp"
#include "umpire/strategy/thread_safe.hpp"
#endif

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Make an Allocator thread safe
 *
 * Using this AllocationStrategy will make the provided allocator thread-safe
 * by syncronizing access to the allocators interface.
 */
class ThreadSafeAllocator : public AllocationStrategy {
 public:
  ThreadSafeAllocator(const std::string& name, int id, Allocator allocator);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

  std::mutex* get_mutex();

 protected:
  strategy::AllocationStrategy* m_allocator;

  std::mutex m_mutex;

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // allocate()/deallocate() forward through v2's thread_safe<v1_backed_memory>,
  // which serializes access with its own internal mutex. m_mutex above is
  // kept (unused for locking allocate/deallocate in this configuration) so
  // get_mutex() -- relied on by umpire::Allocator's own thread-safe path,
  // see src/umpire/Allocator.cpp -- keeps working identically; it still
  // returns a valid, distinct mutex whose locking behavior around
  // allocate()/deallocate() is a superset (Allocator locks around the whole
  // do_allocate()/do_deallocate() call, which now itself also locks the v2
  // delegate's own mutex; this is safe, just slightly more serialization
  // than strictly necessary).
  std::unique_ptr<detail::v1_backed_memory> m_v1_backed_parent;
  std::unique_ptr<thread_safe<detail::v1_backed_memory>> m_delegate;
#endif
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_ThreadSafeAllocator_HPP

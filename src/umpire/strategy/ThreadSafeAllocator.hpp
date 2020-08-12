//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_ThreadSafeAllocator_HPP
#define UMPIRE_ThreadSafeAllocator_HPP

#include <mutex>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

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
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 protected:
  strategy::AllocationStrategy* m_allocator;

  std::mutex m_mutex;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_ThreadSafeAllocator_HPP

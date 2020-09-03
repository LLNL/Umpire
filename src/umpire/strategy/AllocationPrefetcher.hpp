//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationPrefetcher_HPP
#define UMPIRE_AllocationPrefetcher_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Apply the appropriate "PREFETCH" operation to every allocation.
 */
class AllocationPrefetcher : public AllocationStrategy {
 public:
  AllocationPrefetcher(const std::string& name, int id, Allocator allocator,
                       int device_id = 0);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  std::shared_ptr<op::MemoryOperation> m_prefetch_operation;
  strategy::AllocationStrategy* m_allocator;
  int m_device;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_AllocationPrefetcher_HPP

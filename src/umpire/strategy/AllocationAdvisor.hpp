//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AllocationAdvisor_HPP
#define UMPIRE_AllocationAdvisor_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Applies the given MemoryOperation to every allocation.
 *
 * This AllocationStrategy is designed to be used with the following
 * operations:
 *
 * - op::CudaAdviseAccessedByOperation
 * - op::CudaAdvisePreferredLocationOperation
 * - op::CudaAdviseReadMostlyOperation
 *
 * Using this AllocationStrategy when combined with a pool like DynamicPool is
 * a good way to mitigate the overhead of applying the memory advice.
 */
class AllocationAdvisor : public AllocationStrategy {
 public:
  AllocationAdvisor(const std::string& name, int id, Allocator allocator,
                    const std::string& advice_operation, int device_id = 0);

  AllocationAdvisor(const std::string& name, int id, Allocator allocator,
                    const std::string& advice_operation,
                    Allocator accessing_allocator, int device_id = 0);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  std::shared_ptr<op::MemoryOperation> m_advice_operation;

  strategy::AllocationStrategy* m_allocator;

  int m_device;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_AllocationAdvisor_HPP

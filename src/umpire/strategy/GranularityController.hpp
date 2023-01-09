//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_GranularityController_HPP
#define UMPIRE_GranularityController_HPP

#include <memory>

#include "umpire/Allocator.hpp"
#include "umpire/op/MemoryOperation.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

/*!
 *
 * \brief Control granularity of memory coherence for allocaotr
 *
 * For supported memory resources, the GranularityController will cause the 
 * coherency of the memory allocations to be either fine or course grained.
 *
 * Using this AllocationStrategy when combined with a pool like QuickPool is
 * a good way to creat an entire pool with a specific coherency scheme.
 */
class GranularityController : public AllocationStrategy {
 public:
  enum Granularity {
      FineGrainedCoherence = 1
    , CourseGrainedCohorence = 2
  };

  GranularityController(const std::string& name, int id, Allocator allocator, Granularity granularity,
                    int device_id = 0);

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t size) override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  std::shared_ptr<op::MemoryOperation> m_advice_operation;

  strategy::AllocationStrategy* m_allocator;

  int m_device;
  Granularity m_granularity;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_GranularityController_HPP

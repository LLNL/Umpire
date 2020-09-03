//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SlotPool_HPP
#define UMPIRE_SlotPool_HPP

#include <memory>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace strategy {

class SlotPool : public AllocationStrategy {
 public:
  SlotPool(const std::string& name, int id, Allocator allocator,
           std::size_t slots);

  ~SlotPool();

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr) override;

  std::size_t getCurrentSize() const noexcept override;
  std::size_t getHighWatermark() const noexcept override;

  Platform getPlatform() noexcept override;

  MemoryResourceTraits getTraits() const noexcept override;

 private:
  void init();

  void** m_pointers;
  int64_t* m_lengths;

  std::size_t m_current_size;
  std::size_t m_highwatermark;

  std::size_t m_slots;

  strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SlotPool_HPP

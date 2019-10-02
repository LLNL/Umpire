//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_SlotPool_HPP
#define UMPIRE_SlotPool_HPP

#include <memory>
#include <vector>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

class SlotPool :
  public AllocationStrategy
{
  public:
    SlotPool(
      const std::string& name,
      int id,
      std::size_t slots,
      Allocator allocator);

    ~SlotPool();

    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);

    std::size_t getCurrentSize() const noexcept;
    std::size_t getHighWatermark() const noexcept;

    Platform getPlatform() noexcept;
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

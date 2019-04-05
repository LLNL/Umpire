//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
        size_t slots,
        Allocator allocator);

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;

    long getCurrentSize() const noexcept override;
    long getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;
  private:
    void init();

    void** m_pointers;
    size_t* m_lengths;

    long m_current_size;
    long m_highwatermark;

    size_t m_slots;

    strategy::AllocationStrategy* m_allocator;
};

} // end of namespace strategy
} // end namespace umpire

#endif // UMPIRE_SlotPool_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_AlignedAllocator_HPP
#define UMPIRE_AlignedAllocator_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/Allocator.hpp"

namespace umpire {
namespace strategy {

class AlignedAllocator :
  public AllocationStrategy
{
  public:
    AlignedAllocator(
        const std::string& name,
        int id,
        Allocator allocator,
        std::size_t alignment=8); 

    void* allocate(std::size_t bytes) override;
    void deallocate(void* ptr) override;

    std::size_t getCurrentSize() const noexcept override;
    std::size_t getHighWatermark() const noexcept override;

    Platform getPlatform() noexcept override;

    MemoryResourceTraits getTraits() const noexcept override;

  protected:
    strategy::AllocationStrategy* m_allocator;

  private:
    std::size_t m_alignment;
    uintptr_t m_mask;
};

} // end of namespace strategy
} // end of namespace umpire

#endif

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_aligned_allocation_HPP
#define UMPIRE_aligned_allocation_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include <unordered_map>

namespace umpire {
namespace strategy {
namespace mixins {

class AlignedAllocation {
public:
    AlignedAllocation() = delete;
    AlignedAllocation(std::size_t alignment, strategy::AllocationStrategy* strategy);

    //!
    //! \brief Round up the size to be an integral multple of configured
    //!        alignment.
    //!
    //! \returns Size rounded up to be integral multiple of configured
    //!          alignment
    //!
    std::size_t aligned_round_up(std::size_t size);

    //!
    //! \brief Return an allocation of `size` bytes that is aligned on the
    //!        configured alignment boundary.
    //!
    void* aligned_allocate(const std::size_t size);

    //!
    //! \brief Deallocate previously alligned allocation
    //!
    void aligned_deallocate(void* ptr);

protected:
    strategy::AllocationStrategy* m_allocator;

private:
    std::unordered_map<void*, void*> base_pointer_map;
    std::size_t m_alignment;
    std::size_t m_mask;
};

} // namespace mixins
} // namespace strategy
} // namespace umpire

#include "umpire/strategy/mixins/AlignedAllocation.inl"

#endif // UMPIRE_aligned_allocation_HPP

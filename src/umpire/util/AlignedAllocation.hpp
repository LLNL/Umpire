//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_aligned_allocation_HPP
#define UMPIRE_aligned_allocation_HPP

#include <unordered_map>

namespace umpire {
namespace util {

class AlignedAllocation {
public:
    AlignedAllocation() = delete;
    AlignedAllocation(std::size_t alignment);

    //////////////////////////////////////////////////////////////////////////
    /// \brief Round up the size to be an integral multple of configured
    ///        alignment.
    ///
    /// \returns Size rounded up to be integral multiple of configured
    ///          alignment
    //////////////////////////////////////////////////////////////////////////
    std::size_t round_up(std::size_t size);

    //////////////////////////////////////////////////////////////////////////
    /// \brief Adjust given pointer `ptr` and `size` to storage that is
    ///        aligned to the configured `alignment` number of bytes
    ///
    /// This function will also establish a mapping between the base buffer
    /// address and the aligned address (see: `align_destroy`)
    ///
    /// \param size Input:  size of buffer.
    ///             Output: the actual size of storage after alignment
    /// \param ptr Input: pointer to contiguous storage of at least `size`
    ///                   bytes
    ///            Output: pointer that is aligned by the configured
    ///                    alignment number of bytes.
    //////////////////////////////////////////////////////////////////////////
    void align_create(std::size_t& size, void*& ptr);

    //////////////////////////////////////////////////////////////////////////
    /// \brief Return original address that was aligned by `align_create` and
    ///        delete the mapping previously established.
    //////////////////////////////////////////////////////////////////////////
    void* align_destroy(void* ptr);

private:
    std::unordered_map<void*, void*> base_pointer_map;
    std::size_t m_alignment;
    std::size_t m_mask;
};

} // namespace umpire
} // namespace util

#include "umpire/util/AlignedAllocation.inl"

#endif // UMPIRE_aligned_allocation_HPP

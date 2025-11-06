//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_allocation_metadata_HPP
#define UMPIRE_allocation_metadata_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>

#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/FixedMallocPool.hpp"

namespace umpire {

namespace strategy {
class AllocationStrategy;
}

namespace util {

/*!
 * \brief Default alignment for user data after allocation header
 *
 * Using 64-byte alignment to ensure compatibility with:
 * - Modern CPU cache lines (64 bytes)
 * - GPU memory transfers (typically 64 or 128 bytes)
 * - SIMD operations (16, 32, 64 bytes)
 */
constexpr std::size_t allocation_alignment = 64;

/*!
 * \brief Header stored inline before each tracked allocation
 *
 * This header contains a pointer to metadata stored in a memory pool.
 * The header is padded to Alignment bytes to ensure the user data
 * that follows is properly aligned.
 *
 * \tparam Metadata The type of metadata to store (typically AllocationRecord)
 * \tparam Alignment The alignment requirement for user data (default 64 bytes)
 */
template<typename Metadata, std::size_t Alignment = allocation_alignment>
struct alignas(Alignment) allocation_header {
  Metadata* metadata;
  // Padding to Alignment is automatically added by alignas
};

// Verify that the header has the expected size and alignment
static_assert(sizeof(allocation_header<AllocationRecord>) == allocation_alignment,
              "allocation_header must be exactly allocation_alignment bytes");
static_assert(alignof(allocation_header<AllocationRecord>) == allocation_alignment,
              "allocation_header must be aligned to allocation_alignment");

namespace detail {

/*!
 * \brief Get reference to static FixedMallocPool for allocation metadata
 *
 * \return Reference to static pool instance
 */
inline FixedMallocPool& metadata_pool()
{
  static FixedMallocPool pool{sizeof(AllocationRecord)};
  return pool;
}

/*!
 * \brief Get reference to static mutex protecting the metadata pool
 *
 * \return Reference to static mutex
 */
inline std::mutex& metadata_mutex()
{
  static std::mutex mtx;
  return mtx;
}

} // end of namespace detail

/*!
 * \brief Calculate total allocation size including aligned header
 *
 * Similar to std::size(), returns the total size needed for an allocation
 * including the metadata header.
 *
 * \tparam Alignment The alignment requirement (default 64 bytes)
 * \param user_size Size requested by user
 * \return Total size needed (user_size + Alignment bytes)
 */
template<std::size_t Alignment = allocation_alignment>
constexpr std::size_t allocation_size(std::size_t user_size) noexcept
{
  return user_size + Alignment;
}

/*!
 * \brief Get user data pointer from base allocation pointer
 *
 * Similar to std::data(), converts from the base allocation pointer
 * (start of header) to the user data pointer (after header).
 *
 * \tparam Alignment The alignment of the header (default 64 bytes)
 * \param base_ptr Pointer to the start of the allocation (with space for header)
 * \return Pointer to user data (after the header)
 */
template<std::size_t Alignment = allocation_alignment>
constexpr void* allocation_data(void* base_ptr) noexcept
{
  uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
  return reinterpret_cast<void*>(base + Alignment);
}

/*!
 * \brief Get base allocation pointer from user data pointer
 *
 * Inverse of allocation_data(), converts from user data pointer back to
 * the base allocation pointer where the header is stored.
 *
 * \tparam Alignment The alignment of the header (default 64 bytes)
 * \param user_ptr Pointer to user data (after header)
 * \return Pointer to base allocation (start of header)
 */
template<std::size_t Alignment = allocation_alignment>
constexpr void* allocation_base(void* user_ptr) noexcept
{
  uintptr_t user = reinterpret_cast<uintptr_t>(user_ptr);
  return reinterpret_cast<void*>(user - Alignment);
}

/*!
 * \brief Construct allocation header and metadata
 *
 * Similar to std::construct_at(), this function constructs the allocation
 * header and associated metadata in-place. The metadata is allocated from
 * a memory pool and the header stores a pointer to it.
 *
 * \tparam Metadata The type of metadata to store
 * \tparam Alignment The alignment requirement (default 64 bytes)
 * \tparam Args Types of arguments to forward to Metadata constructor
 *
 * \param base_ptr Pointer to the start of the allocation (with space for header)
 * \param args Arguments to forward to the Metadata constructor
 * \return Pointer to user data (after the header)
 */
template<typename Metadata, std::size_t Alignment = allocation_alignment, typename... Args>
void* construct_allocation(void* base_ptr, Args&&... args)
{
  // Allocate metadata from pool
  void* metadata_mem;
  {
    std::lock_guard<std::mutex> lock(detail::metadata_mutex());
    metadata_mem = detail::metadata_pool().allocate();
  }

  // Calculate user pointer
  void* user_ptr = allocation_data<Alignment>(base_ptr);

  // Construct metadata in place
  Metadata* metadata = new (metadata_mem) Metadata(std::forward<Args>(args)...);

  // Write header
  using header_t = allocation_header<Metadata, Alignment>;
  header_t* header = static_cast<header_t*>(base_ptr);
  header->metadata = metadata;

  return user_ptr;
}

/*!
 * \brief Access allocation metadata
 *
 * Similar to std::get(), retrieves a const reference to the allocation metadata
 * associated with the given user pointer.
 *
 * \tparam Metadata The type of metadata stored
 * \tparam Alignment The alignment of the header (default 64 bytes)
 * \param user_ptr Pointer to user data (after header)
 * \return Const reference to the allocation metadata
 */
template<typename Metadata, std::size_t Alignment = allocation_alignment>
const Metadata& allocation_metadata(void* user_ptr)
{
  void* base = allocation_base<Alignment>(user_ptr);
  using header_t = allocation_header<Metadata, Alignment>;
  header_t* header = static_cast<header_t*>(base);
  return *header->metadata;
}

/*!
 * \brief Destruct allocation header and return metadata copy
 *
 * Symmetric counterpart to construct_allocation(), this function destructs
 * the allocation metadata, returns a copy of it, and frees the metadata back
 * to the pool.
 *
 * \tparam Metadata The type of metadata stored
 * \tparam Alignment The alignment of the header (default 64 bytes)
 * \param user_ptr Pointer to user data (after header)
 * \return Pair of {Metadata copy, base pointer for deallocation}
 */
template<typename Metadata, std::size_t Alignment = allocation_alignment>
std::pair<Metadata, void*> destruct_allocation(void* user_ptr)
{
  // Read header
  void* base = allocation_base<Alignment>(user_ptr);
  using header_t = allocation_header<Metadata, Alignment>;
  header_t* header = static_cast<header_t*>(base);
  Metadata* metadata = header->metadata;

  // Copy metadata data before freeing
  Metadata metadata_copy = *metadata;

  // Destroy and free metadata back to pool
  {
    std::lock_guard<std::mutex> lock(detail::metadata_mutex());
    metadata->~Metadata();
    detail::metadata_pool().deallocate(metadata);
  }

  return {metadata_copy, base};
}

/*!
 * \brief Check if this strategy supports header-based allocation tracking
 *
 * Header-based tracking only works with host-accessible memory:
 *  - Host memory
 *  - CUDA unified/managed memory
 *  - HIP managed memory
 *  - SYCL USM shared memory
 *
 * \param strategy AllocationStrategy to check
 * \return true if header-based tracking is supported
 */
bool supportsHeaderIntrospection(strategy::AllocationStrategy* strategy);

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_allocation_metadata_HPP

//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Umpire_HPP
#define UMPIRE_Umpire_HPP

#include <iostream>
#include <string>

#include "camp/camp.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/MemoryResourceRegistry.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/MPI.hpp"
#include "umpire/util/io.hpp"

namespace umpire {

inline void initialize(
#if defined(UMPIRE_ENABLE_MPI)
    MPI_Comm umpire_communicator
#endif
)
{
  static bool initialized = false;

  if (!initialized) {
#if defined(UMPIRE_ENABLE_MPI)
    util::MPI::initialize(umpire_communicator);
#else
    util::MPI::initialize();
#endif

    initialized = true;
  }
}

void finalize();
/*!
 * \brief Allocate memory in the default space, with the default allocator.
 *
 * This method is a convenience wrapper around calls to the ResourceManager to
 * allocate memory in the default MemorySpace.
 *
 * \param size Number of bytes to allocate.
 */
inline void* malloc(std::size_t size)
{
  return ResourceManager::getInstance().getDefaultAllocator().allocate(size);
}

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
/*!
 * \brief Calculate total size including metadata header overhead.
 *
 * When header introspection is enabled, allocations include a 64-byte
 * metadata header. This function calculates the total size needed.
 * Use this when creating FixedPool strategies to account for the header:
 *
 * Example:
 *   // For 512-byte user allocations with header introspection:
 *   auto pool = rm.makeAllocator<FixedPool>(
 *     "my_pool", base_alloc, umpire::get_allocation_size(512));
 *
 * \param user_size The size of user data in bytes
 * \return Total size including header overhead (user_size + 64)
 */
inline constexpr std::size_t get_allocation_size(std::size_t user_size) noexcept
{
  return user_size + 64;  // 64-byte header for metadata
}
#endif

/*!
 * \brief Free any memory allocated with Umpire.
 *
 * This method is a convenience wrapper around calls to the ResourceManager, it
 * can be used to free allocations from any MemorySpace. *
 *
 * \param ptr Address to free.
 */
inline void free(void* ptr)
{
  return ResourceManager::getInstance().deallocate(ptr);
}

inline int get_major_version()
{
  return UMPIRE_VERSION_MAJOR;
}

inline int get_minor_version()
{
  return UMPIRE_VERSION_MINOR;
}

inline int get_patch_version()
{
  return UMPIRE_VERSION_PATCH;
}

inline std::string get_rc_version()
{
  return UMPIRE_VERSION_RC;
}

/*!
 * \brief Print the allocations from a specific allocator in a
 * human-readable format.
 *
 * \param allocator source Allocator.
 * \param os output stream
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, this function only
 *       prints allocations registered in the ResourceManager's map. Normal
 *       header-tracked allocations are NOT included. This function is primarily
 *       useful for debugging untracked allocators or manual registrations in
 *       header introspection mode.
 */
void print_allocator_records(Allocator allocator, std::ostream& os = std::cout);

/*!
 * \brief Returns vector of AllocationRecords created by the allocator.
 *
 * \param allocator source Allocator.
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, this function only
 *       returns allocations registered in the ResourceManager's map. Normal
 *       header-tracked allocations are NOT included.
 */
std::vector<util::AllocationRecord> get_allocator_records(Allocator allocator);

/*!
 * \brief Check whether the right allocation overlaps the left.
 *
 * right will overlap left if the right is greater than left, but less than
 * left+size, and right+size is strictly greater than left+size.
 *
 * \param left Pointer to left allocation
 * \param right Poniter to right allocation
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, both pointers must
 *       be exact pointers returned by allocate(). Interior pointers result in
 *       undefined behavior.
 */
bool pointer_overlaps(void* left, void* right);

/*!
 * \brief Check whether the left allocation contains the right.
 *
 * right is contained by left if right is greater than left, and right+size is
 * greater than left+size.
 *
 * \param left Pointer to left allocation
 * \param right Poniter to right allocation
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, both pointers must
 *       be exact pointers returned by allocate(). Interior pointers result in
 *       undefined behavior.
 */
bool pointer_contains(void* left, void* right);

/*!
 * \brief Check whether or not an Allocator is accessible from a given platform.
 *
 * This function describes which allocators should be accessible
 * from which CAMP platforms. Information on platform/allocator
 * accessibility can be found at
 * <https://umpire.readthedocs.io/en/develop/features/allocator_accessibility.html>
 *
 *\param camp::Platform p
 *\param umpire::Allocator a
 */
bool is_accessible(Platform p, Allocator a);

/*!
 * \brief Get the backtrace associated with the allocation of ptr
 *
 * The string may be empty if backtraces are not enabled.
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, backtraces are only
 *       available for allocations registered in the ResourceManager's map. Normal
 *       header-tracked allocations do not have stored backtraces.
 */
std::string get_backtrace(void* ptr);

/*!
 * \brief Get memory usage of the current process (uses underlying
 * system-dependent calls)
 */
std::size_t get_process_memory_usage();

/*!
 * \brief Get high watermark memory usage of the current process (uses underlying
 * system-dependent calls)
 */
std::size_t get_process_memory_usage_hwm();

/*!
 * \brief Mark an application-specific event string within Umpire life cycle.
 */
void mark_event(const std::string& event);

/*!
 * \brief Get the total umpire memory usage in bytes across all memory resources
 */
std::size_t get_total_bytes_allocated();

/*!
 * \brief Get memory usage of device device_id, using appropriate underlying
 * vendor API.
 */
std::size_t get_device_memory_usage(int device_id);

/*!
 * \brief Get all the leaked (active) allocations associated with allocator.
 *
 * \note When UMPIRE_ENABLE_HEADER_INTROSPECTION is enabled, this function only
 *       detects leaks from allocations registered in the ResourceManager's map.
 *       Header-tracked allocations cannot be detected by this function. For
 *       comprehensive leak detection, use map-based introspection mode.
 */
std::vector<util::AllocationRecord> get_leaked_allocations(Allocator allocator);

/*!
 * \brief Return the default traits for the given allocator string
 */
umpire::MemoryResourceTraits get_default_resource_traits(const std::string& name);

/*!
 * \brief Return the pointer of an allocation for the given allocator and name
 */
void* find_pointer_from_name(Allocator allocator, const std::string& name);

#if defined(UMPIRE_ENABLE_MPI)
/*!
 * \brief Return the MPI communicator for a shared memory allocator.
 *
 * NOTE: Using this function will REQUIRE users to call the
 * cleanup_cached_communicators() function to avoid memory leaks.
 */
MPI_Comm get_communicator_for_allocator(Allocator a, MPI_Comm comm);
void cleanup_cached_communicators();
#endif

void register_external_allocation(void* ptr, util::AllocationRecord record);
util::AllocationRecord deregister_external_allocation(void* ptr);

/*!
 * \brief Returns the Camp resource associated with a particular allocation
 * **This function is used mainly for testing purposes.**
 *
 * \param Umpire allocator which was used to allocate the data
 * \param Pointer which was used for the allocation
 *
 * \return Camp resource associated with the allocation *assuming the Allocator
 * passed in is a ResourceAwarePool strategy and the allocation is either used or pending*
 */
camp::resources::Resource get_resource(Allocator a, void* ptr);

/*!
 * \brief Returns the number of pending chunks associated with a ResourceAwarePool Allocator
 * **This function is used mainly for testing purposes.**
 *
 * \param Umpire ResourceAwarePool allocator
 *
 * \return Number of currently pending chunks in the ResourceAwarePool
 */
std::size_t get_num_pending(Allocator a);

/*!
 * \brief Attempt to coalesce Allocator a, return true if a coalesce was performed.
 *
 * \return True if the Allocator was coalesced.
 */
bool try_coalesce(Allocator a);

/*!
 * \brief Attempt to coalesce Allocator a, throw if a does not support coalescing.
 *
 * \throw umpire::util::Exception if the Allocator doesn't support coalescing.
 */
void coalesce(Allocator a);

} // end of namespace umpire

#endif // UMPIRE_Umpire_HPP

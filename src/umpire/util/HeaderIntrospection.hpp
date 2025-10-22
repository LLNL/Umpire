//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HeaderIntrospection_HPP
#define UMPIRE_HeaderIntrospection_HPP

#include <cstddef>
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
 * \brief Minimal header stored inline before each tracked allocation
 *
 * This 8-byte header contains a pointer to an AllocationRecord stored
 * in a memory pool, enabling O(1) introspection lookups.
 */
struct IntrospectionHeader {
  AllocationRecord* record;
};

namespace detail {

/*!
 * \brief Get reference to static FixedMallocPool for AllocationRecords
 *
 * \return Reference to static pool instance
 */
inline FixedMallocPool& getRecordPool()
{
  static FixedMallocPool pool{sizeof(AllocationRecord)};
  return pool;
}

/*!
 * \brief Get reference to static mutex protecting the record pool
 *
 * \return Reference to static mutex
 */
inline std::mutex& getRecordPoolMutex()
{
  static std::mutex mtx;
  return mtx;
}

} // end of namespace detail

/*!
 * \brief Allocate AllocationRecord from pool, write header, return user pointer
 *
 * \param base_ptr Pointer to the start of the allocation (with space for header)
 * \param size Size of the user allocation (not including header)
 * \param strategy AllocationStrategy that owns this allocation
 * \param name Optional name for the allocation
 *
 * \return Pointer to user data (after the header)
 */
void* insertHeader(void* base_ptr, std::size_t size, strategy::AllocationStrategy* strategy,
                   const std::string& name = "");

/*!
 * \brief Read header and return allocation record
 *
 * \param user_ptr Pointer to user data (after header)
 *
 * \return Pointer to AllocationRecord
 */
AllocationRecord* getRecord(void* user_ptr);

/*!
 * \brief Read header, free record to pool, return record copy and base pointer
 *
 * \param user_ptr Pointer to user data (after header)
 *
 * \return Pair of {AllocationRecord copy, base pointer for deallocation}
 */
std::pair<AllocationRecord, void*> removeHeader(void* user_ptr);

/*!
 * \brief Calculate total allocation size including header
 *
 * \param user_size Size requested by user
 *
 * \return Total size needed (user_size + sizeof(IntrospectionHeader))
 */
constexpr std::size_t getTotalSize(std::size_t user_size)
{
  return user_size + sizeof(IntrospectionHeader);
}

/*!
 * \brief Check if this strategy supports header introspection
 *
 * Header introspection only works with host-accessible memory:
 *  - Host memory
 *  - CUDA unified/managed memory
 *  - HIP managed memory
 *  - SYCL USM shared memory
 *
 * \param strategy AllocationStrategy to check
 *
 * \return true if header introspection is supported
 */
bool supportsHeaderIntrospection(strategy::AllocationStrategy* strategy);

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_HeaderIntrospection_HPP

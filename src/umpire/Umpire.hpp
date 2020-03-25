//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_Umpire_HPP
#define UMPIRE_Umpire_HPP

#include "umpire/config.hpp"

#include "umpire/util/MPI.hpp"
#include "umpire/util/io.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/util/AllocationRecord.hpp"

#include <iostream>

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
inline
void* malloc(std::size_t size)
{
  return ResourceManager::getInstance().getDefaultAllocator().allocate(size);
}

/*!
 * \brief Free any memory allocated with Umpire.
 *
 * This method is a convenience wrapper around calls to the ResourceManager, it
 * can be used to free allocations from any MemorySpace. *
 *
 * \param ptr Address to free.
 */
inline
void free(void* ptr)
{
  return ResourceManager::getInstance().deallocate(ptr);
}

inline
int get_major_version()
{
  return UMPIRE_VERSION_MAJOR;
}

inline
int get_minor_version()
{
  return UMPIRE_VERSION_MINOR;
}

inline
int get_patch_version()
{
  return UMPIRE_VERSION_PATCH;
}

inline
std::string get_rc_version()
{
  return UMPIRE_VERSION_RC;
}

/*!
 * \brief Print the allocations from a specific allocator in a
 * human-readable format.
 *
 * \param allocator source Allocator.
 * \param os output stream
 */
void print_allocator_records(Allocator allocator, std::ostream& os = std::cout);

/*!
 * \brief Returns vector of AllocationRecords created by the allocator.
 *
 * \param allocator source Allocator.
 */
std::vector<util::AllocationRecord> get_allocator_records(Allocator allocator);

bool pointer_overlaps(void* left, void* right);

bool pointer_contains(void* left, void* right);

std::string get_backtrace(void* ptr);

std::size_t get_process_memory_usage();

std::size_t get_device_memory_usage(int device_id)

} // end of namespace umpire

#endif // UMPIRE_Umpire_HPP

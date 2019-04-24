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
#ifndef UMPIRE_Umpire_HPP
#define UMPIRE_Umpire_HPP

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"

#include "umpire/Allocator.hpp"

#include "umpire/util/AllocationRecord.hpp"

#include <iostream>

namespace umpire {

/*!
 * \brief Allocate memory in the default space, with the default allocator.
 *
 * This method is a convenience wrapper around calls to the ResourceManager to
 * allocate memory in the default MemorySpace.
 *
 * \param size Number of bytes to allocate.
 */
inline
void* malloc(size_t size)
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
std::vector<const util::AllocationRecord*> get_allocator_records(Allocator allocator);

} // end of namespace umpire

#endif // UMPIRE_Umpire_HPP

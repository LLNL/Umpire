//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_pointer_utils_HPP
#define UMPIRE_pointer_utils_HPP

#include "umpire/config.hpp"

namespace umpire {

class ResourceManager;

namespace strategy {
class AllocationStrategy;
}

namespace util {

/*!
 * \brief Infer which allocator owns a pointer by querying runtime APIs
 *
 * This function uses platform-specific APIs to determine the memory type
 * of a pointer and returns the corresponding default allocator.
 *
 * For CUDA: uses cudaPointerGetAttributes
 * For HIP: uses hipPointerGetAttributes
 * For host-only: assumes HOST allocator
 *
 * \param ptr Pointer to query
 * \param rm ResourceManager instance (for allocator lookup)
 * \return Allocator strategy if determinable, nullptr otherwise
 */
strategy::AllocationStrategy* inferAllocatorFromPointer(void* ptr, ResourceManager& rm);

} // end of namespace util
} // end of namespace umpire

#endif // UMPIRE_pointer_utils_HPP

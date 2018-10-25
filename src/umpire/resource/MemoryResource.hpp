//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#ifndef UMPIRE_MemoryResource_HPP
#define UMPIRE_MemoryResource_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

namespace umpire {
namespace resource {

/*!
 * \brief Base class to represent the available hardware resources for memory
 * allocation in the system.
 *
 * Objects of this inherit from strategy::AllocationStrategy, allowing them to
 * be used directly.
 */
class MemoryResource :
  public strategy::AllocationStrategy
{
  public:
    /*!
     * \brief Construct a MemoryResource with the given name and id.
     *
     * \param name Name of the MemoryResource.
     * \param id ID of the MemoryResource (must be unique).
     *
     */
    MemoryResource(const std::string& name, int id);

    virtual ~MemoryResource() = default;

    /*!
     * \brief Allocate bytes of memory.
     *
     * This function is pure virtual and must be implemented by the inheriting
     * class.
     *
     * \param bytes Number of bytes to allocate.
     *
     * \return Pointer to start of allocation.
     */
    virtual void* allocate(size_t bytes) = 0;

    /*!
     * \brief Free the memory at ptr.
     *
     * This function is pure virtual and must be implemented by the inheriting
     * class.
     *
     * \param ptr Pointer to free.
     */
    virtual void deallocate(void* ptr) = 0;

    /*!
     * \brief Return the current size of this MemoryResource.
     *
     * This is sum of the sizes of all the tracked allocations. Note that this
     * doesn't ever have to be equal to getHighWatermark.
     *
     * \return current total size of active allocations in this MemoryResource.
     */
    virtual long getCurrentSize() noexcept = 0;

    /*!
     * \brief Return the memory high watermark for this MemoryResource.
     *
     * This is the largest amount of memory allocated by this Allocator. Note
     * that this may be larger than the largest value returned by
     * getCurrentSize.
     *
     * \return Memory high watermark.
     */
    virtual long getHighWatermark() noexcept = 0;


    /*!
     * \brief Get the Platform assocatiated with this MemoryResource.
     *
     * This function is pure virtual and must be implemented by the inheriting
     * class.
     *
     * \return Platform associated with this MemoryResource.
     */
    virtual Platform getPlatform() noexcept = 0;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP

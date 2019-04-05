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
#ifndef UMPIRE_MemoryResource_HPP
#define UMPIRE_MemoryResource_HPP

#include "umpire/strategy/AllocationStrategy.hpp"

#include "umpire/util/MemoryResourceTraits.hpp"

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
    MemoryResource(const std::string& name, int id, MemoryResourceTraits traits);

    ~MemoryResource() override = default;

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
    void* allocate(size_t bytes) override = 0;

    /*!
     * \brief Free the memory at ptr.
     *
     * This function is pure virtual and must be implemented by the inheriting
     * class.
     *
     * \param ptr Pointer to free.
     */
    void deallocate(void* ptr) override = 0;

    /*!
     * \brief Return the current size of this MemoryResource.
     *
     * This is sum of the sizes of all the tracked allocations. Note that this
     * doesn't ever have to be equal to getHighWatermark.
     *
     * \return current total size of active allocations in this MemoryResource.
     */
    long getCurrentSize() const noexcept override = 0;

    /*!
     * \brief Return the memory high watermark for this MemoryResource.
     *
     * This is the largest amount of memory allocated by this Allocator. Note
     * that this may be larger than the largest value returned by
     * getCurrentSize.
     *
     * \return Memory high watermark.
     */
    long getHighWatermark() const noexcept override = 0;


    /*!
     * \brief Get the Platform assocatiated with this MemoryResource.
     *
     * This function is pure virtual and must be implemented by the inheriting
     * class.
     *
     * \return Platform associated with this MemoryResource.
     */
    Platform getPlatform() noexcept override = 0;

    MemoryResourceTraits getTraits();
  protected:
    MemoryResourceTraits m_traits;
};

} // end of namespace strategy
} // end of namespace umpire

#endif // UMPIRE_MemoryResource_HPP
